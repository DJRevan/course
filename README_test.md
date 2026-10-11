# Checkpoint 1.1 - Wikipedia Capstone Baseline

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

---

# Checkpoint 2.1 — Baseline Retrieval

This second checkpoint adds a retrieval layer to the Wikipedia baseline. Instead of answering from model memory alone, the system retrieves relevant articles first and then grounds the answer in those documents.

## Goal

The goal is to build a simple baseline retriever for the Wikipedia corpus and test whether it can find the most relevant articles for queries such as:

- a specific person, place, or event,
- a fact that needs one article,
- a question that spans multiple articles,
- a query whose wording differs from the article text.

## Retrieval strategies used

The starter script implements three retrieval approaches:

- BM25 keyword matching
- vector retrieval using embeddings
- hybrid retrieval combining BM25 and vector scores

These mirror the core design patterns used in the course labs and provide a simple baseline for a more advanced Wikipedia search system.

## Files for this checkpoint

- `capstone_checkpoint_2_1_baseline_retrieval_starter.py` — baseline retrieval workflow
- `checkpoint_2_1_retrieval.log` — saved retrieval outputs and answers
- `Wikipedia/` — article corpus used for retrieval
- `chroma_baseline/` — local vector store used for semantic search

## Run it

```bash
python capstone_checkpoint_2_1_baseline_retrieval_starter.py
```

The script will:

1. load the Wikipedia corpus,
2. retrieve candidate documents with BM25, vector, and hybrid methods,
3. pass the top results to the LLM with a grounding instruction,
4. print and log the retrieved documents and generated answer.

## Evaluation focus

For each query, review:

- which documents were retrieved,
- whether the most relevant article appears in the top results,
- whether the answer is supported by the retrieved text,
- where the baseline still fails (weak keyword matches, noisy retrieval, or poor multi-document coverage).

This checkpoint is the transition from a no-retrieval baseline to a grounded retrieval system for the Wikipedia scenario.

---

# Checkpoint 3.1 — Evaluation and Baseline Diagnosis

This checkpoint adds a structured evaluation layer to the Wikipedia retrieval system. After building a baseline retriever, the goal is to measure how well the system answers questions, diagnose failure modes, and check whether the evaluation process can reliably detect bad answers.

## Goal

The goal is to evaluate whether the retrieval-augmented system is actually useful for the Wikipedia scenario. The script tests questions and grading notes using an LLM judge, then measures pass/fail performance for the baseline.

## What this checkpoint covers

- evaluation design for retrieval-based QA,
- use of an LLM judge to score responses,
- test cases that include both easy and difficult questions,
- framework validation to confirm the evaluator can catch manipulated answers.

## Files for this checkpoint

- `capstone_checkpoint_3_1_evaluation_starter.py` — evaluation workflow and baseline diagnosis
- `checkpoint_3_1_evaluation.log` — saved evaluation outputs
- `testInputs.json` and `testInputs_variants.json` — optional evaluation inputs for broader testing

## Run it

```bash
python capstone_checkpoint_3_1_evaluation_starter.py
```

The script will:

1. run the evaluation set,
2. retrieve relevant Wikipedia-style documents,
3. generate answers from the retrieved context,
4. pass answers through a judge model,
5. print the pass/fail results and save them to the log file.

## Evaluation focus

Review the results to determine:

- which questions the baseline handles well,
- which questions fail because retrieval is weak,
- whether the judge is strict enough to catch unsupported answers,
- how the system could be improved in later checkpoints.

This checkpoint is where the project shifts from building the retrieval system to systematically measuring its quality and diagnosing weaknesses.

---

# Checkpoint 3.1 Judge Summary

`capstone_checkpoint_3_1_judge.py` is the evaluation companion for the baseline retrieval system. It focuses on the idea that a model answer should be judged against explicit grading notes, not just whether it sounds fluent or confident.

## What the judge does

The script:

- retrieves the most relevant documents for a question,
- asks the model to answer using only the retrieved context,
- compares the answer against a set of grading notes,
- uses a second LLM call to return a strict `pass` or `fail` verdict.

This creates a lightweight evaluation loop for the Wikipedia pipeline and makes it possible to diagnose which questions are working and which ones still fail.

## Why it matters

The judge is useful because many incorrect answers look convincing even when they are unsupported. A good evaluator checks whether the answer actually satisfies the task requirements and whether it is grounded in the retrieved material.

## Typical use

The evaluation set includes example questions such as:

- factual questions about a person or place,
- questions requiring a specific article or quote,
- questions where the model should say it cannot establish a claim,
- questions that should fail if the answer is fabricated or unsupported.

The judge then reports the system's pass rate and helps identify where the retrieval + answer pipeline still needs work.

## Practical takeaway

This script is the foundation for later, more systematic evaluation work. It shows how to turn a retrieval system into a measurable pipeline by checking whether the output matches the task requirements and not just whether it is verbose or plausible.

---

# Checkpoint 4.1 — Multi-Step Retrieval and Reasoning

`capstone_checkpoint_4_1_multi_step.py` extends the Wikipedia retrieval pipeline by moving beyond single-step retrieval toward a multi-hop reasoning workflow. The goal is to answer questions that require more than one retrieval step, more than one document, or a chain of inferences before producing a grounded answer.

## What this checkpoint adds

This checkpoint introduces a more realistic reasoning pattern for capstone questions:

- retrieve initial candidate documents,
- expand the search based on the most relevant paths,
- gather supporting context from multiple sources,
- answer a question that depends on synthesis rather than a single lookup.

It is designed for questions that are harder than simple fact lookup, such as:

- multi-hop questions that connect two concepts,
- questions requiring comparison across articles,
- queries where the answer depends on a chain of evidence,
- edge cases where a system should avoid unsupported assumptions.

## Why it matters

A basic retriever can often find the right article for a direct fact query, but many real-world tasks require reasoning across multiple evidence sources. This checkpoint tests whether the system can do that while staying grounded in the retrieved documents instead of guessing.

## Typical workflow

The script:

1. loads the Wikipedia corpus,
2. retrieves relevant documents using a hybrid or multi-step retrieval strategy,
3. builds a richer context from the top candidates,
4. asks the LLM to answer using only the retrieved information,
5. evaluates the result with a correctness metric and logs the output.

## Evaluation focus

In this checkpoint, the most important questions are:

- did the system retrieve the right supporting documents,
- did it combine them correctly,
- did it avoid hallucinating missing information,
- did the final answer reflect the actual evidence rather than model priors.

This checkpoint is the step from a strong retrieval baseline to a more reasoning-oriented Wikipedia assistant.

---

# Checkpoint 5.1 — Agentic Retrieval Updates

`capstone_checkpoint_5_1_agentic_updates.py` introduces an agentic update to the Wikipedia question-answering pipeline. Instead of relying on a single static retrieval pass, the system decides whether it has enough evidence and, if not, issues a follow-up search query to gather the missing information.

## What this checkpoint adds

This checkpoint moves the project toward an agentic workflow:

- inspect the current evidence,
- decide whether the answer is fully supported,
- generate a targeted follow-up query when evidence is incomplete,
- iteratively refine retrieval before producing the final answer.

This is important for questions that require more than one retrieval step, where the first result set is incomplete or partly relevant.

## Why it matters

A static retrieval pipeline can miss the exact evidence needed for a good answer. The agentic version is more flexible: it checks whether the current retrieval is sufficient, then asks for the missing fact instead of guessing or stopping too early.

## Typical workflow

The script:

1. retrieves an initial set of relevant documents,
2. evaluates whether the evidence is sufficient,
3. formulates a focused next query when needed,
4. repeats the cycle until the evidence is adequate or a max step limit is reached,
5. answers using only the accumulated supporting context.

## Evaluation focus

The key question in this checkpoint is whether the agent can improve retrieval quality without drifting into unsupported conclusions. In practice, this means checking whether the follow-up queries are targeted, whether the evidence becomes more complete, and whether the final answer remains grounded in the retrieved text.

This checkpoint marks the transition from a fixed retrieval pipeline to a more adaptive, decision-making agent for the Wikipedia capstone system.

---

# Checkpoint 6.1 — Security and Cost Monitoring

`capstone_checkpoint_6_1_security_cost.py` focuses on making the Wikipedia system safer and more operationally realistic. It adds the monitoring layer needed to understand how expensive and risky an agentic retrieval system can become when it makes multiple LLM calls and processes more evidence.

## What this checkpoint adds

This checkpoint emphasizes two practical concerns:

- security and guardrails around model behavior,
- cost and latency tracking for repeated API calls.

The script measures how the system uses tokens, models, and inference calls during retrieval and answer generation, so the performance is not judged only by output quality but also by operating cost.

## Why it matters

More capable systems often make more calls, retrieve more context, and consume more tokens. Without monitoring, it is easy to build a system that works on a few examples but is too expensive or too brittle for real use. This checkpoint adds the infrastructure for measuring those trade-offs.

## Typical workflow

The script:

1. records API-call timing and token usage,
2. tracks the number of retrieval and model calls per question,
3. lets the system run through a realistic evaluation set,
4. reports approximate costs and latency alongside pass/fail behavior.

## Evaluation focus

The key question here is: how much quality are we getting for the cost? This checkpoint encourages comparing system performance against the number of calls, latency, and token usage so the project can balance correctness, security, and efficiency.

This checkpoint is the bridge between a working agentic system and a production-minded system that is designed to be safe, measurable, and cost-aware.

---

# Checkpoint 7.1 — Safe and Efficient Agent

`capstone_checkpoint_7_1_safe_efficient_agent.py` is the final capstone checkpoint focused on making the Wikipedia agent both safer and more efficient. It builds on the earlier agentic and monitoring work by adding safeguards that reduce noisy retrieval, prevent low-quality evidence from driving the answer, and avoid unnecessary extra calls.

## What this checkpoint adds

This checkpoint introduces practical controls for real-world agent behavior:

- relevance checks before continuing with a retrieval path,
- guardrails for weak or off-topic evidence,
- early stopping when the answer is not well supported,
- efficiency controls to avoid unnecessary model calls and redundant searches.

The goal is to improve answer quality while keeping the system more targeted and less wasteful.

## Why it matters

A powerful agent can become unreliable if it keeps searching without improving evidence quality or if it answers based on weak matches. This checkpoint addresses that by requiring the agent to judge whether the retrieved evidence is truly relevant and sufficient before moving forward.

## Typical workflow

The script:

1. retrieves candidate passages,
2. estimates whether the retrieved evidence is relevant to the question,
3. decides whether more retrieval is needed,
4. stops early when the evidence is poor or the answer is unsupported,
5. produces a final answer using only evidence that passes the relevance and safety checks.

## Evaluation focus

The key question is whether the system can remain grounded, avoid low-confidence guesses, and use fewer unnecessary calls without sacrificing answer quality. In other words, it balances correctness, safety, and efficiency.

This final checkpoint represents the mature version of the capstone system: a retrieval agent that is not only capable, but also disciplined, economical, and better aligned with trustworthy AI behavior.

---

# Project Roadmap

1. 1.1 — establish the no-retrieval baseline and expose model weaknesses.
2. 2.1 — add a basic retrieval system over the Wikipedia corpus.
3. 3.1 — evaluate retrieval quality and diagnose failures with an LLM judge.
4. 4.1 — move to multi-step reasoning across multiple evidence sources.
5. 5.1 — add agentic follow-up retrieval when the initial evidence is incomplete.
6. 6.1 — track security, cost, and operational trade-offs.
7. 7.1 — finalize a safe, efficient, grounded retrieval agent.
