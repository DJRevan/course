r"""Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation (starter).
Jupytext-style cell markers (# %% / # %% [markdown]) — runnable as a
plain script AND openable as cells in VS Code / PyCharm / Jupytext.
"""

# %% [markdown]
# # Capstone Checkpoint 2.1 — Retrieval Strategy Design and Baseline Implementation
# **MO-LLM Module 2 / Required Capstone Checkpoint (120 minutes)**
#
# ## What this checkpoint is
#
# In Checkpoint 1.1, you showed that a plain LLM can't reliably answer questions about
# your corpus. Now, you will **add retrieval**: Design a retrieval strategy for your scenario
# and build a **baseline retrieval system** that finds the most relevant documents for
# a query, so the model can ground its answers in them.
#
# This mirrors the Module 2 labs — keyword (BM25), vector (semantic), and hybrid
# retrieval — applied to your own capstone corpus. The graded deliverable is the completed 
# Capstone Checkpoint 2.1 worksheet, which includes your written responses and evidence of your 
# retrieval system implementation and testing. This script provides a small working example of 
# baseline retrieval. Use it to understand the retrieval workflow, then adapt the code to implement 
# and test a baseline retriever using your selected capstone dataset.
#
# **Learning outcomes (Module 2):**
# 1. Design a retrieval strategy appropriate for a given dataset and query type.
# 2. Implement and test a baseline retrieval system using structured and/or semantic
#    approaches.

# %% [markdown]
# ## Step 1 — Keep your capstone scenario
#
# Use the **same scenario** you chose in Checkpoint 1.1.
#
# | Scenario | Corpus | Retrieval considerations |
# |---|---|---|
# | **Research Paper Navigator** | ~150 research-paper PDFs (`Labs/CapstoneDatasets/ResearchPapers/`) | long documents; you'll likely chunk them; questions often name a specific paper or compare papers. |
# | **Wikipedia Retrieval Engine** | ~2,400 Wikipedia HTML articles (`Labs/CapstoneDatasets/Wikipedia/`) | many short-to-medium articles; questions name a figure/place or span several articles. |
#
# A good baseline is keyword (BM25), semantic (embeddings + vector search), or a
# hybrid of both — exactly what you built in Labs 1.2–2.2.

# %% [markdown]
# ## Setup (~5 min)
#
# 1. **Python 3.11 or 3.12**
# 2. `pip install langchain-openai langchain-core python-dotenv`
# 3. Use the OpenRouter API key provided for this program. This checkpoint uses
#  the `openai/gpt-5.4-mini` model, with usage covered by the course credits. (this uses the paid gpt-5.4-mini chat model — covered by your course credits — and a keyword retriever, no embeddings).
# 4. Create a `.env` file next to this script: `OPENROUTER_API_KEY=sk-or-v1-...`
#
# This runs on a tiny built-in sample corpus, so you do not need to prepare your own
# dataset. It still requires an OpenRouter API key to run the LLM (it is not offline or
# free of API calls). Your real baseline (over your full corpus) is what you describe in
# the writeup.

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

# %this part is the same as before, reaching the LLM model
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "openai/gpt-5.4-mini"  # latest small OpenAI model, fast; covered by course credits
TEMPERATURE = 0.2
TOP_K = 3
LOG_PATH = Path.cwd() / "checkpoint_2_1_retrieval.log"
CORPUS_DIR = Path(__file__).resolve().parent / "Wikipedia"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_baseline"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "Wikipedia"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)

# this segments checks whether I have the API or not, and if not, it exits with a message telling me to get one.
def check_api_key() -> str:
    load_dotenv()
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Use the OpenRouter API key "
            "provided for this course, put it in a .env file next to this "
            "script, and rerun."
        )
    return key

def make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )

def log(label: str, text: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {label}\n{text}\n{'-' * 72}\n")


# %% [markdown]
# ## A tiny sample corpus (stands in for your real one)
#
# There are six short "documents" on distinct topics so a baseline retriever has something to
# discriminate between. Your real corpus is the PDFs/articles in `Labs/CapstoneDatasets/`,
# which came with the course in Module 1. Point your code at your local copy of that
# folder, and update the path if your checkout puts it elsewhere.


# %%
class _HTMLTextExtractor(HTMLParser):
    """Extract readable text from a raw Wikipedia HTML file."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        cleaned = re.sub(r"\s+", " ", data).strip()
        if cleaned:
            self._parts.append(cleaned)

    def get_text(self) -> str:
        return " ".join(self._parts)


def extract_html_text(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        raw_html = fh.read()
    parser = _HTMLTextExtractor()
    parser.feed(raw_html)
    parser.close()
    return parser.get_text()


def load_wikipedia_docs(corpus_dir: Path = CORPUS_DIR) -> list[dict[str, str]]:
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        raise FileNotFoundError(f"Wikipedia corpus folder not found: {corpus_dir}")

    docs: list[dict[str, str]] = []
    for html_file in sorted(corpus_dir.glob("*.html")):
        text = extract_html_text(html_file)
        if text.strip():
            docs.append({
                "id": html_file.stem,
                "text": text,
                "source": html_file.name,
            })

    if not docs:
        raise ValueError(f"No HTML documents were found in {corpus_dir}")
    return docs


DOCS = load_wikipedia_docs()
DOC_BY_ID = {d["id"]: d for d in DOCS}


# %% [markdown]
# ## Step 2 — The baseline retriever (provided)
#
# A simple keyword-overlap retriever scores each document by how many query words
# it shares, and returns the top-k. This is the smallest possible baseline (a stand-in
# for the BM25 / vector / hybrid retriever you built in the Module 2 labs). The
# `answer` function then asks the LLM using only the retrieved documents.

# %%
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_retrieve(query: str, docs: list[dict[str, str]], k: int = TOP_K) -> list[tuple[str, float]]:
    """Keyword retrieval using BM25 over the Wikipedia HTML corpus."""
    if BM25Okapi is None:
        raise RuntimeError("Install rank-bm25 to use the BM25 retriever: pip install rank-bm25")

    corpus = [_tokens(d["text"]) for d in docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokens(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked:
        if score > 0:
            results.append((docs[idx]["id"], float(score)))
        if len(results) >= k:
            break
    return results


def build_vector_db(docs: list[dict[str, str]]) -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=check_api_key(),
        base_url=OPENROUTER_BASE_URL,
    )
    # Persist the index next to the script so repeated runs reuse it.
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        [Document(page_content=d["text"], metadata={"source": d["source"], "id": d["id"]}) for d in docs],
        embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def vector_retrieve(query: str, db: Chroma, k: int = TOP_K) -> list[tuple[str, float]]:
    """Semantic retrieval using embeddings and cosine similarity."""
    results = db.similarity_search_with_score(query, k=k)
    ranked = []
    for doc, score in results:
        doc_id = doc.metadata.get("id") or doc.metadata.get("source") or "unknown"
        ranked.append((doc_id, float(score)))
    return ranked


def retrieve(query: str, docs: list[dict[str, str]] | None = None, db: Chroma | None = None,
             strategy: str = "bm25", k: int = TOP_K) -> list[tuple[str, float]]:
    """Dispatch to the BM25 or vector retriever."""
    docs = docs or DOCS
    if strategy == "bm25":
        return bm25_retrieve(query, docs, k=k)
    if strategy == "vector":
        if db is None:
            db = build_vector_db(docs)
        return vector_retrieve(query, db, k=k)
    raise ValueError(f"Unknown retrieval strategy: {strategy}")


def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content


# %% [markdown]
# ## Step 3 — Your representative queries (TODO)
#
# Submission item #2 asks for **3-5 representative queries** for your scenario and the
# results your system retrieves for each. Write those queries here. Some good ones include questions that:
#
# - Are answerable from **one** document (tests precision),
# - Need **several** documents (tests recall / aggregation),
# - Have wording that **differs** from the document's wording (i.e., tests whether
#   keyword vs. semantic retrieval matters for your corpus)
#
# Return a list of 3-5 query strings.

# %%
def my_representative_queries() -> list[str]:
    
     return [
        "Which film is described as the turning point that brought Ana de Armas major international recognition?",

        "What creature is portrayed as a defining symbol of Tasmania’s natural environment?",

        "How does the Uranium article characterize the element’s industrial importance, and what does the Tasmania article "
        "mention about the region’s mineral deposits?",

        "What thematic parallels or contrasts can be drawn between Clive Barker’s creative style and the narrative motifs "
        "described in Sailor Moon?",

        "How does Sailor Moon contribute to the broader tradition of magical‑hero storytelling, based on its article?"
        
    ]

  #  raise NotImplementedError("my_representative_queries() — see the TODO above.")


# %% [markdown]
# ## Step 4 — Run the baseline and capture the evidence
#
# This runs each query through the baseline retriever and the LLM, printing the
# retrieved document ids/scores and the grounded answer, and logging everything to
# `checkpoint_2_1_retrieval.log`. The retrieved documents from the output are the rest of the evidence for
# submission item #2.

# %%
def run() -> None:
    llm = make_llm()
    queries = my_representative_queries()
    vector_db = build_vector_db(DOCS)
    print(f"Checkpoint 2.1 — baseline retrieval  |  scenario: {SCENARIO}\n")
    for i, query in enumerate(queries, 1):
        print("=" * 72)
        print(f"QUERY {i}: {query}")

        bm25_hits = retrieve(query, docs=DOCS, strategy="bm25", k=TOP_K)
        vector_hits = retrieve(query, docs=DOCS, db=vector_db, strategy="vector", k=TOP_K)

        print(f"  BM25 retrieved: {bm25_hits}")
        print(f"  Vector retrieved: {vector_hits}")

        best_hits = bm25_hits if bm25_hits else vector_hits
        if not best_hits:
            print("  (nothing matched — note this in your writeup)")
            continue
        ans = answer(llm, query, [doc_id for doc_id, _ in best_hits])
        print(f"  answer: {ans}\n")
        log(f"QUERY {i}: {query}", f"bm25={bm25_hits}\nvector={vector_hits}\nanswer={ans}")
    print("=" * 72)
    print("Done. Use the retrieved document results above as evidence in your writeup, and "
          "describe your REAL baseline and vector baseline (over your full corpus) in the submission.")


run()

# %% [markdown]
# ## Step 5 — Your written submission (the graded deliverable)
#
# Use your completed retrieval implementation and test results to complete the Capstone Checkpoint 2.1
# worksheet. In the worksheet, you will document your retrieval approach, provide evidence that your 
# system is functioning, include 3–5 representative queries and retrieved results, and reflect on where
# your approach performs well and where it struggles.  
 
# Save your completed Python file in the appropriate checkpoint folder in your GitHub repository. 
# Upload the completed worksheet only to the learning platform as your graded submission.
