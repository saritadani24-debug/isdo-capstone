"""
ISDO - Knowledge Base loader and retrieval test.

1. Reads every .md file in data/kb/
2. Splits each article into chunks at '## ' headings
3. Stores all chunks in the ChromaDB collection 'isdo_kb'
4. Runs 4 sample queries and prints the best matching article + confidence

Uses only the chromadb package (and its built-in default embedding model,
all-MiniLM-L6-v2, which is downloaded once on first run).

Run from anywhere, e.g. from the project folder:
    python .\\labs\\C1\\kb_setup.py
"""

import re
import sys
from pathlib import Path

import chromadb

# ---------------------------------------------------------------- config ---
def find_project_root() -> Path:
    """Walk up from this script's folder until a 'data/kb' folder is found,
    so the script works from any sub-folder (e.g. labs/C1/)."""
    script_dir = Path(__file__).resolve().parent
    for folder in [script_dir, *script_dir.parents]:
        if (folder / "data" / "kb").is_dir():
            return folder
    sys.exit(f"Could not find a 'data/kb' folder in {script_dir} or any parent folder")


BASE_DIR = find_project_root()
KB_DIR = BASE_DIR / "data" / "kb"
DB_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "VPN keeps saying authentication failed after I changed my password",
    "SAP login fails with DBCON_FAIL for our whole finance team",
    "Outlook on my phone is not syncing new emails",
    "Everyone on the 3rd floor lost network, the switch looks dead",
]

# Windows consoles can choke on characters like '→' in the articles
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


# ------------------------------------------------------------- chunking ---
def split_into_chunks(md_path: Path) -> list[dict]:
    """Split one markdown article at level-2 ('## ') headings.

    The text before the first '## ' (title + category/tags) becomes an
    'Overview' chunk. '### ' sub-headings stay inside their parent chunk.
    Each chunk is prefixed with the article title so it carries context.
    """
    text = md_path.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else md_path.stem

    # Split right before every line that starts with exactly '## '
    parts = re.split(r"(?m)^(?=##\s)", text)

    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            section = part.splitlines()[0][3:].strip()
        else:
            section = "Overview"
        chunks.append(
            {
                "id": f"{md_path.stem}::{len(chunks):02d}",
                "document": f"{title}\n\n{part}",
                "metadata": {
                    "article": md_path.name,
                    "title": title,
                    "section": section,
                    "chunk_index": len(chunks),
                },
            }
        )
    return chunks


# --------------------------------------------------------------- loading ---
def build_collection():
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        sys.exit(f"No .md files found in {KB_DIR}")

    all_chunks = []
    for f in md_files:
        chunks = split_into_chunks(f)
        all_chunks.extend(chunks)
        print(f"  {f.name:<28} -> {len(chunks)} chunks")

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Rebuild from scratch so re-runs never leave stale chunks behind
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine distance -> 0..1 confidence
    )
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["document"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
    )
    print(f"\nStored {collection.count()} chunks from {len(md_files)} articles "
          f"in collection '{COLLECTION_NAME}'.\n")
    return collection


# --------------------------------------------------------------- testing ---
def run_queries(collection):
    print("=" * 78)
    print("SAMPLE QUERIES")
    print("=" * 78)
    for i, query in enumerate(SAMPLE_QUERIES, 1):
        res = collection.query(
            query_texts=[query],
            n_results=3,
            include=["metadatas", "distances"],
        )
        top_meta = res["metadatas"][0][0]
        top_dist = res["distances"][0][0]
        confidence = max(0.0, 1.0 - top_dist)  # cosine similarity

        print(f"\nQ{i}: {query}")
        print(f"  Best match : {top_meta['article']}  ({top_meta['title']})")
        print(f"  Section    : {top_meta['section']}")
        print(f"  Confidence : {confidence:.2%}")

        runners_up = [
            f"{m['article']} [{m['section']}] {1 - d:.0%}"
            for m, d in zip(res["metadatas"][0][1:], res["distances"][0][1:])
        ]
        print(f"  Next best  : {' | '.join(runners_up)}")


if __name__ == "__main__":
    print(f"Reading articles from {KB_DIR}\n")
    kb = build_collection()
    run_queries(kb)
