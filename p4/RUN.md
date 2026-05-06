# ResearchFlow — Project Notes & Run Guide

## Overview

ResearchFlow is a multi-agent research assistant built with LangGraph, LangChain, AWS Bedrock, and Pinecone. The system orchestrates multiple specialized agents under a Supervisor graph to answer complex research questions using Adaptive RAG, self-refinement, fact-checking, and HITL (Human-in-the-Loop) review.

This document supplements the original `README.md` specification and documents implementation decisions, evaluation outcomes, runtime instructions, and development tradeoffs.

---

# Quick Start

## 1. Create Virtual Environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

Create a `.env` file in the project root.

Example:

```env
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET
AWS_DEFAULT_REGION=us-east-1

PINECONE_API_KEY=YOUR_KEY
PINECONE_INDEX_NAME=researchflow

BEDROCK_CHAT_MODEL_ID=model
BEDROCK_EMBEDDING_MODEL_ID=model
```

---

## 4. Run the Supervisor Graph

```bash
python3.11 -m agents.supervisor
```

---

## 5. Launch the Mission Control Dashboard

```bash
python3.11 -m streamlit run dashboard.py
```

---

# Mission Control Dashboard

The project includes a custom Streamlit “Mission Control” dashboard for visualizing agent orchestration in real time.

Dashboard features include:

- Active agent highlighting
- Confidence score tracking
- Retrieved source panel
- Scratchpad / reasoning trace
- Retry iteration tracking
- Checkpoint timeline
- HITL approval workflow
- Raw graph state debugger

The dashboard was created to improve observability and make the LangGraph orchestration easier to demonstrate during presentations.

---

# Corpus Used

## Primary Corpus

The primary corpus consisted of:

- Philosophy educational PDFs
- Philosophy glossary documents
- Utilitarianism lecture/transcript materials
- General ethics and political philosophy reference texts
- Transcripts from the YouTube channel CrashCourse's series on philosophy

Example files:

- `GlossaryOfPhilosophy.pdf`
- `Understanding-the-Major-Branches-of-Philosophy-2.pdf`
- `Utilitarianism_Crash_Course_Philosophy_#36.txt`

## Why This Corpus Was Chosen

The philosophy domain was selected because:

1. It contains nuanced, multi-hop reasoning problems.
2. It allows meaningful fact-checking workflows.
3. It exposes hallucination risks in LLM-generated explanations.
4. It provides opportunities for self-refinement and critique loops.
5. Philosophical concepts often require synthesis across multiple sources.

The dataset was intentionally small during development to:
- reduce Pinecone costs
- speed up iteration/debugging
- simplify evaluation

---

# Architecture Highlights

## Multi-Agent Workflow

ResearchFlow uses a LangGraph Supervisor architecture containing:

- Planner Node
- Retriever Agent
- Analyst Agent
- Fact-Checker Agent
- Critique Node

The graph supports:
- conditional routing
- retry loops
- HITL interrupts
- checkpoint-based time travel
- sliding window message management

---

## Adaptive RAG Features

Implemented RAG enhancements include:

- Pinecone semantic search
- metadata-aware retrieval
- relevance scoring
- context compression
- re-ranking
- scratchpad observability
- sliding-window memory trimming

---

# Sliding Window Message Management

The Supervisor graph includes sliding-window message trimming to manage token growth during long-running sessions.

Implementation details:

- Recent conversational messages are stored in graph state.
- Older messages are trimmed automatically.
- Trim events are logged into the scratchpad.
- This prevents prompt explosion during retry loops and HITL workflows.

---

# RAGAS Evaluation Results

## Final Scores

| Metric | Score |
|---|---|
| Faithfulness | 0.81 |
| Answer Relevancy | 0.86 |
| Context Precision | 0.74 |

## Commentary

### Faithfulness
The system generally remained grounded in retrieved context, especially after introducing fact-check retries and critique loops.

### Answer Relevancy
The Planner + Analyst combination produced strong answers for conceptual and multi-hop philosophy questions.

### Context Precision
This was the weakest metric. Some retrieved chunks were only loosely related to the generated answer, especially in the fact-check namespace.

---

# Future Improvements

## Retrieval Improvements

The biggest next improvement would be:

- stronger chunking strategies
- better reranking
- hybrid search (dense + sparse)
- domain-specific embeddings
- query rewriting

---

## Fact-Checking Improvements

The fact-checker occasionally returned broad philosophy references instead of highly targeted evidence.

Future improvements:
- dedicated fact-check corpus
- stricter metadata filtering
- citation-level verification
- claim decomposition

---

## UI Improvements

Potential dashboard upgrades:

- live streaming agent execution
- graph visualization
- retry heatmaps
- confidence-over-time charts
- token usage tracking

---

# Deviations From Original Spec

## Claude Haiku Instead of Sonnet

The implementation used Claude Haiku in some development/testing scenarios instead of Sonnet.

### Reason
- lower cost
- faster iteration speed
- reduced latency during debugging

---

## Streaming Responses

Streaming was partially implemented but not fully integrated into the dashboard UI.

### Reason
Streamlit integration and graph orchestration complexity increased debugging overhead during the limited project timeline.

---

## Smaller Corpus During Development

A smaller development corpus was used than originally envisioned.

### Reason
- reduced Pinecone costs
- faster indexing
- quicker iteration cycles

---

# Cost Notes

## Estimated Development Cost

Approximate total development cost:

| Service | Estimated Cost |
|---|---|
| AWS Bedrock | ~$8–15 (free tier first) |
| Pinecone | ~$0–5 |
| Streamlit | Free |
| LangSmith | Free tier |
| Total | ~$10–20 |

---

## Biggest Cost Drivers

### Bedrock Inference
The largest cost came from:
- repeated Analyst synthesis
- retry loops
- fact-checking passes

### Embedding Generation
Titan embeddings generated moderate cost during ingestion.

### Pinecone
Costs remained minimal because:
- serverless index was used
- small corpus size
- lightweight development dataset

---

# Observability

The project includes:

- structured logging
- scratchpad tracing
- LangGraph checkpoint history
- HITL state inspection
- Mission Control Dashboard
- retry observability

---

# Testing

The project includes:

- mocked retriever tests
- mocked analyst tests
- supervisor routing tests
- retry loop validation
- HITL workflow testing

---

# Final Notes

ResearchFlow evolved beyond a traditional RAG chatbot into a stateful, self-refining multi-agent orchestration system.

The strongest outcomes of the project were:

- multi-agent coordination
- critique-driven retries
- HITL escalation
- checkpoint time travel
- observability tooling
- dashboard visualization

The project demonstrates production-oriented AI system design patterns rather than only basic retrieval-augmented generation.