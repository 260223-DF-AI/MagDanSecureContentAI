# Document Corpus

Place your research documents here. The ingestion pipeline (`scripts/ingest.py`) will process all files in this directory.

## Supported Formats

- `.pdf` — PDF documents (parsed with pypdf)
- `.txt` — Plain text files

## Tips

- Include at least **10–15 documents** to build a meaningful corpus for retrieval evaluation.
- Consider using a themed dataset (e.g., research papers on a specific topic, company documentation, technical manuals) so that multi-hop questions can be constructed.
- If you want to test the **fact-check namespace**, place a separate set of reference/authority documents and ingest them into the `fact-check-sources` namespace:

```bash
python scripts/ingest.py --input-dir ./data/fact-check-sources --namespace fact-check-sources
```

## Dev Notes
Our corpus consists of transcripts from the YouTube channel CrashCourse's series on philosophy. Transcripts are stored locally as .txt files. Future project extension may aim to integrate API transcript retrieval. Limiting to this single dataset allows us to properly assess the effectiveness of our RAG application and easily manage the preprocessing step.
