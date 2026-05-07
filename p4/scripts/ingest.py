"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus --namespace primary-corpus
     - OR -
    python -m scripts.ingest --input-dir ./data/corpus --namespace primary-corpus
    python -m scripts.ingest --input-dir ./data/fact_checker --namespace fact-check-sources
"""

import argparse
import logging
import os
import re
import time

from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logs.log_config import setup_logging
from pinecone import Pinecone

setup_logging()
logger = logging.getLogger("researchflow.ingest")


def parse_args() -> argparse.Namespace:
    """Parse ingestion CLI arguments."""
    logger.info("Parsing CLI arguments...")
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to directory containing PDF/text documents.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="primary-corpus",
        help="Pinecone namespace to upsert into.",
    )
    return parser.parse_args()


def load_documents(input_dir: str) -> list:
    """
    Load and return raw documents from the input directory.

    - Support PDF files (e.g., using pypdf or LangChain's PyPDFLoader).
    - Support plain text files.
    - Return a list of Document objects with content and metadata
      (source filename, page number).
    """
    docs = []
    logger.info(f"Loading documents from: {input_dir}")

    # Loop through dir
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            doc = os.path.join(root, filename)
            name = filename.replace("_", ":")
            name = name.replace(" ", "_")

            try:
                logger.debug(f"Loaded document: {doc}")

                # Check file type
                ext = os.path.splitext(doc)[1].lower()
                if ext not in [".txt", ".pdf"]:
                    logger.warning(f"Skipping unsupported filetype: {doc}")
                    continue

                # -------------------------
                # Handle TXT files
                # -------------------------
                if ext == ".txt":
                    with open(doc, "r", encoding="utf-8") as f:
                        text = f.read()

                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": doc,
                                "filename": name,
                                "page_number": 1,
                            },
                        )
                    )
                    continue
                # -------------------------
                # Handle PDF files
                # -------------------------
                if ext == ".pdf":
                    loader = PyMuPDFLoader(doc)
                    pdf_pages = loader.load()

                    for page in pdf_pages:
                        page.metadata["source"] = doc
                        page.metadata["filename"] = name
                    docs.extend(pdf_pages)
                    continue

            except FileNotFoundError as e:
                logger.error(f"File not found: {doc} — {e}")
                continue
            except Exception as e:
                logger.error(f"Error loading document {doc}: {e}")
                continue

    logger.info(f"Loaded {len(docs)} documents.")
    return docs


def clean_text(text: str) -> str:
    """Reduce noisy data in documents."""
    text = re.sub(r"^\s*\d+/\d+/\d+.*$", "", text, flags=re.MULTILINE)  # remove dates
    text = re.sub(r"https?://\S+", "", text)  # remove URLs
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4}.*", "", text)  # remove pdf artifacts
    # remove citation info
    text = re.sub(r"CITATION INFORMATION.*?(?=\n\n)", "", text, flags=re.DOTALL)
    text = re.sub(r"Page \d+ of \d+", "", text)  # remove page numbers
    text = re.sub(r"\s+", " ", text)  # normalize whitespace
    return text.strip()


def chunk_documents(documents: list) -> list:
    """
    Split documents into smaller chunks for embedding.

    - Use RecursiveCharacterTextSplitter or sentence-level splitting.
    - Attach chunk metadata (chunk_id, source, page_number, timestamp).
    """
    chunks = []
    logger.info("Chunking documents...")

    # Load document
    for doc in documents:
        logger.debug(f"Chunking file: {doc}")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,  # transcript chunking: 1200
                chunk_overlap=80,  # transcript chunking: 200
                separators=["\n\n", "\n", ". "],  # add in transcript chunking: " ", ""
            )
            # Split text
            cleaned = clean_text(doc.page_content)
            texts = text_splitter.split_text(cleaned)
            # Create list of Documents
            texts = [
                add_chunk_metadata(
                    d,
                    doc.metadata,
                    {
                        "source": doc.metadata.get("filename", "unknown"),
                        "chunk_num": n,
                        "id": f"{doc.metadata['filename']}_chunk_{n}",
                        "timestamp": int(time.time()),
                    },
                )
                for n, d in enumerate(texts)
            ]  # Add chunk metadata

            chunks.extend(texts)
        except Exception as e:
            logger.error(f"Error chunking {doc}: {e}")
            continue

    logger.info(f"Generated {len(chunks)} total chunks.")
    return chunks


def add_chunk_metadata(
    chunk_text: str, base_metadata: dict, new_metadata: dict
) -> Document:
    """
    Create a new Document for a chunk of text.

    Merges file-level and chunk-level metadata.
    """
    try:
        merged = {**base_metadata, **new_metadata}
        return Document(page_content=chunk_text, metadata=merged)

    except Exception as e:
        logger.error(
            f"add_chunk_metadata() failed. chunk_text={chunk_text[:50]}..., "
            f"base_metadata={base_metadata}, new_metadata={new_metadata}, error={e}"
        )
        raise


def generate_embeddings(chunks: list) -> list:
    """
    Generate vector embeddings for document chunks in batches.

    - Use Sentence Transformers (e.g., all-MiniLM-L6-v2)
      or Bedrock Titan Embeddings.
    - Process in batches for efficiency (see W5 Monday — batch embedding).
    """
    embedder = BedrockEmbeddings(
        model_id=os.getenv("BEDROCK_EMBEDDING_MODEL_ID"),
        region_name=os.getenv("AWS_REGION"),
    )
    BATCH_SIZE = 32
    embeddings = []
    logger.info("Generating embeddings...")

    # Extract text from Document objs
    texts = [chunk.page_content for chunk in chunks]

    # Process in batches
    for i in range(0, len(texts), BATCH_SIZE):
        try:
            logger.debug(f"Embedding batch {i}–{i + BATCH_SIZE}")
            batch = texts[i : i + BATCH_SIZE]
            batch_embeddings = embedder.embed_documents(batch)
            embeddings.extend(
                batch_embeddings
            )  # extend() preserves correct list nesting
        except Exception as e:
            logger.error(f"Error embedding batch {i}–{i + BATCH_SIZE}: {e}")
            raise

    logger.info(f"Generated {len(embeddings)} embeddings.")
    return embeddings


def upsert_to_pinecone(chunks: list, embeddings: list, namespace: str) -> None:
    """
    Upsert embedding vectors and metadata into the Pinecone index.

    - Initialize the Pinecone client using env vars.
    - Upsert vectors with rich metadata into the specified namespace.
    """
    pc = Pinecone(os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

    if pc is None or index is None:
        logger.error("Missing Pinecone environment variables.")
        raise ValueError(
            "Error retrieving Pinecone .env variables: ('PINECONE_API_KEY',"
            "'PINECONE_INDEX_NAME'). Verify the correct values are in your .env file."
        )

    vectors = []
    logger.info(f"Upserting {len(embeddings)} vectors into namespace '{namespace}'...")

    for i, chunk in enumerate(chunks):
        vectors.append(
            {
                "id": chunk.metadata["id"],
                "values": embeddings[i],
                "metadata": {**chunk.metadata, "text": chunk.page_content},
            }
        )
    logger.debug(f"Sample vector obj: {vectors[0]}")

    BATCH_SIZE = 100
    logger.info(f"Upserting in batches of {BATCH_SIZE}...")
    try:
        for start in range(0, len(vectors), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch = vectors[start:end]

            index.upsert(vectors=batch, namespace=namespace)
            logger.info(
                f"Upserted batch {start // BATCH_SIZE + 1} ({len(batch)} vectors)."
            )

        logger.info("Pinecone upsert completed.")
    except Exception as e:
        logger.error(f"Error upserting vectors to Pinecone: {e}")
        raise


def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    logger.info("============== Starting ingest.py ==============")
    args = parse_args()

    documents = load_documents(args.input_dir)
    chunks = chunk_documents(documents)
    embeddings = generate_embeddings(chunks)
    upsert_to_pinecone(chunks, embeddings, args.namespace)

    logger.info(f"✅ Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
