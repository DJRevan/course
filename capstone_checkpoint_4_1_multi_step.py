r"""This file is the continuation of "Capstone Checkpoint 3.1".


"""

# %%
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    BM25Okapi = None

# %this part is the same as before, reaching the LLM model
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
JUDGE_MODEL = os.getenv("OPENROUTER_JUDGE_MODEL", LLM_MODEL)
TEMPERATURE = 0.2
TOP_K = 3
HOP_CANDIDATES = 5
LOG_PATH = Path.cwd() / "checkpoint_4_1_retrieval.log"
CORPUS_DIR = Path(__file__).resolve().parent / "Wikipedia_text_test"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_baseline"

# === SET THIS to the scenario you chose in Checkpoint 1.1 ===
SCENARIO = "Wikipedia"   # "research_papers" or "wikipedia"

ANSWER_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "documents, and quote from them where you can. If the documents do not contain "
    "the answer, say so rather than guessing."
)
# judge system addition: it will check if the answer satisfies the grading notes, and return pass or fail.

JUDGE_SYSTEM = (
    "You are a strict evaluator. You are given an ANSWER and GRADING NOTES. "
    "Reply with exactly one word: pass if the answer satisfies the grading notes, "
    "otherwise reply fail."
)

# the function itself
def make_ragas_judge():
    return llm_factory(
        JUDGE_MODEL,
        client=OpenAI(api_key=check_api_key(), base_url=OPENROUTER_BASE_URL),
    )


correctness_metric = DiscreteMetric(
    name="grounded_correctness",
    prompt=(
        "You are grading a retrieval-augmented answer. Return 'pass' only when "
        "the response answers the question's central point, is consistent with "
        "the grading notes, and is supported by the retrieved context. Return "
        "'fail' if it is incorrect, unsupported, hallucinates, or misses the "
        "central point. Minor wording differences are acceptable.\n"
        "Question: {question}\n"
        "Response: {response}\n"
        "Grading Notes: {grading_notes}\n"
        "Retrieved Context: {context}"
    ),
    allowed_values=["pass", "fail"],
)

# the questions are related to the ones defined in checkpoint 2.1, and the grading notes are the criteria for passing or failing the answer.
def my_eval_set() -> list[dict[str, str]]:
    return [
        {
            "question": (
                "Which film brought Ana de Armas major international recognition?"
            ),
            "grading_notes": (
                "The answer must identify the film described as the turning point "
                "for Ana de Armas's international recognition."
            ),
        },
        {
            "question": (
                "What creature is described as a defining symbol of Tasmania's "
                "natural environment?"
            ),
            "grading_notes": (
                "The answer must identify the creature associated with Tasmania's "
                "natural environment."
            ),
        },
        {
            "question": (
                "How does the Uranium article describe the element's industrial "
                "importance?"
            ),
            "grading_notes": (
                "The answer must explain that uranium has important industrial or "
                "energy-related uses."
            ),
        },
        {
            "question": (
                "How does Sailor Moon contribute to magical-hero storytelling?"
            ),
            "grading_notes": (
                "The answer must describe Sailor Moon's contribution to the "
                "magical-hero or magical-girl tradition."
            ),
        },
        {
            "question": "What is the connection between two unrelated articles?",
            "grading_notes": (
                "The answer should state that the provided documents do not contain "
                "enough information to establish the connection."
            ),
        },
    ]

def run_evaluation() -> None:
    llm = make_llm()
    judge_llm = make_ragas_judge()
    eval_set = my_eval_set()
    vector_db = build_vector_db(DOCS)
    passes = 0

    print(f"Checkpoint 4.1 evaluation | scenario: {SCENARIO}\n")

    for i, item in enumerate(eval_set, 1):
        hits = retrieve(
            item["question"],
            docs=DOCS,
            db=vector_db,
            strategy="multi_hop",
            k=TOP_K,
            bm25_weight=0.4,
        )

        if hits:
            doc_ids = [doc_id for doc_id, _ in hits]
            generated_answer = answer(llm, item["question"], doc_ids)
            retrieved_context = "\n\n".join(
                f"[{doc_id}] {DOC_BY_ID[doc_id]['text']}"
                for doc_id in doc_ids
                if doc_id in DOC_BY_ID
            )
        else:
            generated_answer = "(no documents retrieved)"
            retrieved_context = "(no documents retrieved)"

        verdict = correctness_metric.score(
            llm=judge_llm,
            question=item["question"],
            response=generated_answer,
            grading_notes=item["grading_notes"],
            context=retrieved_context,
        ).value

        passes += verdict == "pass"

        print("=" * 72)
        print(f"Q{i}: {item['question']}")
        print(f"Retrieved: {hits}")
        print(f"Verdict: {verdict.upper()}")
        print(f"Answer: {generated_answer}")

        log(
            f"Q{i}: {item['question']}",
            f"retrieved={hits}\n"
            f"verdict={verdict}\n"
            f"answer={generated_answer}",
        )

    print("=" * 72)
    print(f"Baseline pass rate: {passes}/{len(eval_set)}")



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



# %%
def extract_text(file_path: Path) -> str:
    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def load_wikipedia_docs(corpus_dir: Path = CORPUS_DIR) -> list[dict[str, str]]:
    if not corpus_dir.exists() or not corpus_dir.is_dir():
        raise FileNotFoundError(f"Wikipedia corpus folder not found: {corpus_dir}")

    docs: list[dict[str, str]] = []
    for text_file in sorted(corpus_dir.glob("*.txt")):
        text = extract_text(text_file)
        if text.strip():
            docs.append({
                "id": text_file.stem,
                "text": text,
                "source": text_file.name,
            })

    if not docs:
        raise ValueError(f"No TXT documents were found in {corpus_dir}")
    return docs


DOCS = load_wikipedia_docs()
DOC_BY_ID = {d["id"]: d for d in DOCS}


# %%
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "how", "in", "into", "is", "it", "its", "of", "on",
    "or", "that", "the", "their", "these", "this", "those", "to", "was",
    "were", "what", "when", "where", "which", "who", "with", "between",
    "does", "do", "not", "about", "two",
}


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    ]

def bm25_retrieve(query: str, docs: list[dict[str, str]], k: int = TOP_K) -> list[tuple[str, float]]:
    """Keyword retrieval using BM25 over the Wikipedia text corpus."""
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
    # Reuse the persisted index so repeated evaluations do not re-embed the corpus.
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if (CHROMA_DIR / "chroma.sqlite3").exists():
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
        )

    return Chroma.from_documents(
        [Document(page_content=d["text"], metadata={"source": d["source"], "id": d["id"]}) for d in docs],
        embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def vector_retrieve(query: str, db: Chroma, k: int = TOP_K) -> list[tuple[str, float]]:
    """Semantic retrieval using embeddings and cosine similarity.

    Chroma's similarity_search_with_score() returns a distance, where lower values
    mean more similar. For hybrid weighting we convert that to a similarity score
    in the same direction as BM25 by using 1.0 - distance.
    """
    results = db.similarity_search_with_score(query, k=k)
    ranked = []
    for doc, score in results:
        doc_id = doc.metadata.get("id") or doc.metadata.get("source") or "unknown"
        ranked.append((doc_id, float(score)))
    return ranked


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Map raw scores to a 0..1 range so BM25 and vector scores can be combined."""
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {doc_id: 1.0 for doc_id in scores}
    return {
        doc_id: (value - min_score) / (max_score - min_score)
        for doc_id, value in scores.items()
    }

def hybrid_retrieve(query: str, docs: list[dict[str, str]], db: Chroma, k: int = TOP_K,
                   bm25_weight: float = 0.4, vector_weight: float | None = None) -> list[tuple[str, float]]:
    """Combine BM25 and vector retrieval scores with explicit weights.

    The two weights should sum to 1.0. If vector_weight is omitted, it is derived as
    1.0 - bm25_weight so the default behavior stays the same as before.

    Examples:
    - bm25_weight=0.7, vector_weight=0.3 -> favor keyword matching
    - bm25_weight=0.4, vector_weight=0.6 -> favor semantic matching
    """
    if vector_weight is None:
        vector_weight = 1.0 - bm25_weight

    if not 0.0 <= bm25_weight <= 1.0:
        raise ValueError("bm25_weight must be between 0.0 and 1.0")
    if not 0.0 <= vector_weight <= 1.0:
        raise ValueError("vector_weight must be between 0.0 and 1.0")
    if abs((bm25_weight + vector_weight) - 1.0) > 1e-9:
        raise ValueError("bm25_weight and vector_weight must sum to 1.0")

    bm25_hits = bm25_retrieve(query, docs, k=max(k * 3, 10))
    vector_hits = vector_retrieve(query, db, k=max(k * 3, 10))

    bm25_map = {doc_id: score for doc_id, score in bm25_hits}
    vector_map = {doc_id: max(0.0, 1.0 - score) for doc_id, score in vector_hits}

    all_ids = set(bm25_map) | set(vector_map)
    if not all_ids:
        return []

    bm25_norm = _normalize_scores(bm25_map)
    vector_norm = _normalize_scores(vector_map)

    combined = []
    for doc_id in all_ids:
        score = (
            bm25_weight * bm25_norm.get(doc_id, 0.0)
            + vector_weight * vector_norm.get(doc_id, 0.0)
        )
        combined.append((doc_id, float(score)))

    return sorted(combined, key=lambda item: item[1], reverse=True)[:k]


def _expand_query(query: str, seed_ids: list[str], docs_by_id: dict[str, dict[str, str]]) -> str:
    """Build a second-hop query from salient terms in first-hop documents."""
    query_terms = set(_tokens(query))
    term_counts: dict[str, int] = {}
    for doc_id in seed_ids:
        text = docs_by_id[doc_id]["text"]
        for token in _tokens(text):
            if token not in query_terms and len(token) > 3:
                term_counts[token] = term_counts.get(token, 0) + 1

    salient_terms = sorted(
        term_counts,
        key=lambda token: (-term_counts[token], token),
    )[:8]
    return f"{query} {' '.join(salient_terms)}".strip()


def multi_hop_retrieve(
    query: str,
    docs: list[dict[str, str]],
    db: Chroma | None = None,
    k: int = TOP_K,
    bm25_weight: float = 0.4,
) -> list[tuple[str, float]]:
    """Retrieve seed documents, expand the query, then retrieve a second hop."""
    docs_by_id = {doc["id"]: doc for doc in docs}
    first_hop = (
        hybrid_retrieve(query, docs, db, k=HOP_CANDIDATES, bm25_weight=bm25_weight)
        if db is not None
        else bm25_retrieve(query, docs, k=HOP_CANDIDATES)
    )
    if not first_hop:
        return []

    expanded_query = _expand_query(
        query,
        [doc_id for doc_id, _ in first_hop],
        docs_by_id,
    )
    second_hop = (
        hybrid_retrieve(expanded_query, docs, db, k=HOP_CANDIDATES, bm25_weight=bm25_weight)
        if db is not None
        else bm25_retrieve(expanded_query, docs, k=HOP_CANDIDATES)
    )

    scores: dict[str, float] = {}
    for rank, (doc_id, score) in enumerate(first_hop):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + 1)
    for rank, (doc_id, score) in enumerate(second_hop):
        scores[doc_id] = scores.get(doc_id, 0.0) + 0.5 / (rank + 1)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]


def retrieve(query: str, docs: list[dict[str, str]] | None = None, db: Chroma | None = None,
             strategy: str = "bm25", k: int = TOP_K,
             bm25_weight: float = 0.4, vector_weight: float | None = None) -> list[tuple[str, float]]:
    """Dispatch to the BM25, vector, or hybrid retriever."""
    docs = docs or DOCS
    if strategy == "bm25":
        return bm25_retrieve(query, docs, k=k)
    if strategy == "vector":
        if db is None:
            db = build_vector_db(docs)
        return vector_retrieve(query, db, k=k)
    if strategy == "hybrid":
        if db is None:
            db = build_vector_db(docs)
        return hybrid_retrieve(query, docs, db=db, k=k,
                               bm25_weight=bm25_weight, vector_weight=vector_weight)
    if strategy == "multi_hop":
        return multi_hop_retrieve(
            query,
            docs,
            db=db,
            k=k,
            bm25_weight=bm25_weight,
        )
    raise ValueError(f"Unknown retrieval strategy: {strategy}")


def answer(llm: ChatOpenAI, query: str, doc_ids: list[str]) -> str:
    context = "\n\n".join(f"[{i}] {DOC_BY_ID[i]['text']}" for i in doc_ids if i in DOC_BY_ID)
    messages = [
        SystemMessage(content=ANSWER_SYSTEM),
        HumanMessage(content=f"Documents:\n{context}\n\nQuestion: {query}"),
    ]
    return llm.invoke(messages).content

# .\.venv\Scripts\python.exe .\capstone_checkpoint_4_1_multi_step.py --offline

def run_offline_check() -> None:
    """Check local corpus loading and BM25 retrieval without API calls."""
    eval_set = my_eval_set()
    print(f"Offline check | loaded {len(DOCS)} documents from {CORPUS_DIR}")

    for i, item in enumerate(eval_set, 1):
        hits = retrieve(
            item["question"],
            docs=DOCS,
            strategy="multi_hop",
            k=TOP_K,
        )
        print(f"Q{i}: {item['question']}")
        print(f"  Multi-hop hits: {hits}")

    print("Offline check passed: corpus loading and BM25 retrieval work.")


def main() -> None:
    if "--offline" in os.sys.argv:
        run_offline_check()
    else:
        run_evaluation()


if __name__ == "__main__":
    main()



