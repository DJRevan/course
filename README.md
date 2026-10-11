# Wikipedia Capstone Baseline

This project evaluates whether a plain LLM can answer questions about a Wikipedia corpus without retrieval. The goal is to show that a model can sound confident while still hallucinating facts, fabricating quotes, or missing evidence.

## Purpose

This checkpoint tests the baseline for a Wikipedia-based retrieval system. It answers one key question:

> Do we need retrieval before the model can reliably answer questions about Wikipedia articles?

The script intentionally uses no retrieval and no document parsing. It is designed to expose model weaknesses before building the real system.

## Setup

Required packages:

```bash
pip install langchain-openai langchain-core python-dotenv
```

Create a `.env` file in the project directory:

```env
OPENROUTER_API_KEY=your_key_here
```

Then run:

```bash
python capstone_checkpoint_1_1_baseline_starter.py
```

## Scenario configuration

Set the scenario in the script:

```python
SCENARIO = "wikipedia"
```

This script sends a set of Wikipedia-specific prompts to the model and logs each prompt/response pair to `checkpoint_1_1_responses.log`.

## What to look for

Review each answer and note whether the model:

- hallucinates a fact,
- gives a generic answer without evidence,
- hedges instead of answering,
- fabricates a quote,
- or answers correctly only because of prior knowledge.

These failures are the evidence for the written checkpoint analysis.

## Deliverable

The graded output is a short analysis explaining why Wikipedia retrieval is necessary, what types of questions require grounding, and how missing or irrelevant information affects trust and reliability.
