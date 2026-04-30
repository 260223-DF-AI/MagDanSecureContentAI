"""
ResearchFlow — Document Ingestion Pipeline

Reads PDF/text files from an input directory, chunks them,
generates embeddings, and upserts them into a Pinecone index.

Usage:
    python scripts/ingest.py --input-dir ./data/corpus --namespace primary-corpus
"""
import os
import argparse
import logging
from pinecone import Pinecone
from dotenv import load_dotenv
from log_config import setup_logging
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    for doc in input_dir:
        try:
            logger.debug(f"Loaded document: {doc}")
            
            # Check file type
            ext = os.path.splitext(doc)[1].lower()
            if ext not in [".txt", ".pdf"]:
                logger.warning(f"Skipping unsupported filetype: {doc}")
                continue
            
            old_metadata = doc.metadata
            print("TESTING TO SEE METADATA CONTENTS: ",old_metadata) # temp to see contents
            docs.append(Document(page_content=doc.page_content, meta_data=old_metadata)) # Add document to list
        except FileNotFoundError as e:
            logger.error(f"File not found: {doc} — {e}")
            continue
        except Exception as e:
            logger.error(f"Error loading document {doc}: {e}")
            continue
    
    logger.info(f"Loaded {len(docs)} documents.")
    return docs


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
            with open(doc) as f:
                text = f.read()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            # Split text
            texts = text_splitter.split_text(text)
            # Create list of Documents
            texts = [
                add_chunk_metadata(d, {"source": doc, "chunk_num": n, "id": f"{doc}_chunk_{n}"}) for n, d in enumerate(texts)
            ] # Add chunk metadata
            
            chunks.extend(texts)
        except Exception as e:
            logger.error(f"Error chunking {doc}: {e}")
            continue
    
    logger.info(f"Generated {len(chunks)} total chunks.")
    return chunks


def add_chunk_metadata(doc: str, new_metadata: dict) -> Document:
    """
    Helper function to add metadata to a Document object

    Args:
        doc (str): text that needs metadata
        new_metadata (dict): new metadata to add to given text

    Returns:
        Document: obj with new metadata
    """
    old_metadata = doc.metadata
    new_metadata = {**old_metadata, **new_metadata}
    return Document(page_content = doc.page_content, metadata = new_metadata)


def generate_embeddings(chunks: list) -> list:
    """
    Generate vector embeddings for document chunks in batches.

    - Use Sentence Transformers (e.g., all-MiniLM-L6-v2)
      or Bedrock Titan Embeddings.
    - Process in batches for efficiency (see W5 Monday — batch embedding).
    """
    embedder = BedrockEmbeddings(
        model_id = "amazon.titan-embed-text-v1",
        region_name = os.getenv("AWS_REGION")
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
            batch = texts[i:i + BATCH_SIZE]
            batch_embeddings = embedder.embed_documents(batch)
            embeddings.extend(batch_embeddings) # extend() preserves correct list nesting
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
        raise ValueError("Error retrieving Pinecone .env variables: ('PINECONE_API_KEY', 'PINECONE_INDEX_NAME'). Verify the correct values are in your .env file.")
    
    vectors = []
    logger.info(f"Upserting {len(embeddings)} vectors into namespace '{namespace}'...")

    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": chunk.metadata["id"],
            "values": embeddings[i],
            "metadata": {
                **chunk.metadata,
                "text": chunk.page_content
            }
        })
    logger.debug(f"Prepared {len(vectors)} vectors for upsert.")
    
    try:
        index.upsert(vectors=vectors, namespace=namespace)
        logger.info("Pinecone upsert completed.")
    except Exception as e:
        logger.error(f"Error upserting vectors to Pinecone: {e}")
        raise


def main() -> None:
    """Orchestrate the full ingestion pipeline."""
    load_dotenv()
    args = parse_args()

    documents = load_documents(args.input_dir)
    chunks = chunk_documents(documents)
    embeddings = generate_embeddings(chunks)
    upsert_to_pinecone(chunks, embeddings, args.namespace)

    logger.info(f"✅ Ingested {len(chunks)} chunks into namespace '{args.namespace}'.")


if __name__ == "__main__":
    main()
