"""
Turn a folder of documents into a corpus.json for DocPilot.

Usage:
    python ingest.py --source docs/            # ingest a whole folder
    python ingest.py --source manual.pdf       # single file
    python ingest.py --source https://...      # web page
    python ingest.py --source docs/ --append   # add to existing corpus

Supported: .pdf, .txt, .md, .rst, .docx
Output: data/corpus.json  (or --output <path>)

Optional deps:
    pip install pdfplumber          # for PDFs
    pip install python-docx         # for .docx
    pip install requests beautifulsoup4  # for URLs
"""

import argparse
import json
import re
from pathlib import Path

try:
    import pdfplumber
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

try:
    import docx as _docx
    _HAS_DOCX = True
except ImportError:
    _HAS_DOCX = False


# Chunk tuning. These work well for technical documentation.
# If your documents are very short (e.g. FAQ entries), lower MIN_CHUNK.
CHUNK_WORDS  = 350   # target words per passage
OVERLAP_WORDS = 40   # overlap between adjacent chunks so context isn't cut off
MIN_CHUNK    = 60    # passages shorter than this get merged with the next one


def _chunk_text(text: str, source: str) -> list[dict]:
    """Paragraph-aware chunking with word-level overlap."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    passages = []
    current_words: list[str] = []

    for para in paras:
        words = para.split()
        if len(current_words) + len(words) > CHUNK_WORDS and len(current_words) >= MIN_CHUNK:
            passages.append(" ".join(current_words))
            # keep a tail of the previous chunk so we don't lose context mid-sentence
            current_words = current_words[-OVERLAP_WORDS:] + words
        else:
            current_words.extend(words)

    if len(current_words) >= MIN_CHUNK:
        passages.append(" ".join(current_words))
    elif passages:
        # too-short tail: merge into previous passage
        passages[-1] += " " + " ".join(current_words)

    stem = Path(source).stem if not source.startswith("http") else source
    return [{"id": i, "source": source, "topic": stem, "text": p} for i, p in enumerate(passages)]


def _ingest_pdf(path: Path) -> list[dict]:
    if not _HAS_PDF:
        raise ImportError("PDF support requires pdfplumber: pip install pdfplumber")
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
    return _chunk_text("\n\n".join(pages), path.name)


def _ingest_txt(path: Path) -> list[dict]:
    return _chunk_text(path.read_text(encoding="utf-8", errors="replace"), path.name)


def _ingest_docx(path: Path) -> list[dict]:
    if not _HAS_DOCX:
        raise ImportError("DOCX support requires python-docx: pip install python-docx")
    doc = _docx.Document(str(path))
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return _chunk_text(text, path.name)


def _ingest_url(url: str) -> list[dict]:
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("URL support requires: pip install requests beautifulsoup4")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["nav", "footer", "script", "style", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n\n", strip=True)
    return _chunk_text(text, url)


def ingest_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _ingest_pdf(path)
    elif suffix in (".txt", ".md", ".rst"):
        return _ingest_txt(path)
    elif suffix == ".docx":
        return _ingest_docx(path)
    else:
        print(f"  skipping {path.name} -- unsupported format")
        return []


def ingest_source(source: str) -> list[dict]:
    if source.startswith("http://") or source.startswith("https://"):
        return _ingest_url(source)
    p = Path(source)
    if p.is_dir():
        all_passages = []
        supported = {".pdf", ".txt", ".md", ".rst", ".docx"}
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in supported:
                print(f"  {f.name} ...", end=" ", flush=True)
                chunks = ingest_file(f)
                print(f"{len(chunks)} passages")
                all_passages.extend(chunks)
        return all_passages
    elif p.is_file():
        return ingest_file(p)
    else:
        raise FileNotFoundError(f"Not found: {source}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into DocPilot corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source",  required=True, help="folder, file path, or URL")
    parser.add_argument("--output",  default="data/corpus.json")
    parser.add_argument("--append",  action="store_true",
                        help="append to existing corpus.json instead of replacing it")
    args = parser.parse_args()

    print(f"Ingesting: {args.source}")
    new_passages = ingest_source(args.source)

    if not new_passages:
        print("Nothing ingested.")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.append and out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        offset = max(p["id"] for p in existing) + 1
        for p in new_passages:
            p["id"] += offset
        combined = existing + new_passages
        print(f"Appended {len(new_passages)} passages (total: {len(combined)})")
    else:
        combined = new_passages
        print(f"Ingested {len(combined)} passages")

    out.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
