"""Build data/train.jsonl from the open-source AWS documentation.

AWS user guides are published as Markdown in the `awsdocs/*` GitHub repositories
(`doc_source/` folder). This script shallow-clones them, cleans the Markdown and
writes a JSONL file in the {"text": ..., "source": ...} format expected by
src/embedding.py.

Usage:
    python scripts/build_corpus.py
"""
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "data" / "train.jsonl"

# GitHub repo -> (branch containing doc_source/, service name).
# These repos are archived: the default branch is sometimes emptied, hence the
# explicit branch. The service name is stored in the metadata.
REPOS = {
    "amazon-s3-user-guide": ("master", "Amazon S3"),
    "iam-user-guide": ("main", "AWS IAM"),
    "amazon-vpc-user-guide": ("master", "Amazon VPC"),
}

MIN_CHARS = 200  # skip near-empty pages (mere tables of contents)


def clone(repo, branch, dest):
    url = f"https://github.com/awsdocs/{repo}.git"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", "-b", branch, url, str(dest)],
        check=True,
    )


def clean_markdown(text):
    """Remove formatting artifacts that do not help retrieval."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)   # HTML comments
    text = re.sub(r"\{\.[^}]*\}", "", text)                    # {.class} attributes
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)           # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)       # [text](link) -> text
    text = re.sub(r"<[^>]+>", "", text)                        # HTML tags
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|<>])", r"\1", text)  # Markdown escapes
    text = re.sub(r"\n{3,}", "\n\n", text)                     # repeated blank lines
    return text.strip()


def iter_documents(repo, service, repo_dir):
    doc_dir = repo_dir / "doc_source"
    if not doc_dir.exists():
        print(f"  ! {repo}: no doc_source folder, skipped")
        return
    for md_file in sorted(doc_dir.glob("*.md")):
        text = clean_markdown(md_file.read_text(encoding="utf-8", errors="ignore"))
        if len(text) < MIN_CHARS:
            continue
        yield {
            "text": text,
            "source": f"{service} - {md_file.stem}",
        }


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    total = 0
    tmp_root = Path(tempfile.mkdtemp())
    try:
        with open(OUTPUT, "w", encoding="utf-8") as out:
            for repo, (branch, service) in REPOS.items():
                print(f"Cloning {repo} ({branch})...")
                repo_dir = tmp_root / repo
                clone(repo, branch, repo_dir)
                count = 0
                for doc in iter_documents(repo, service, repo_dir):
                    out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    count += 1
                print(f"  {count} pages kept")
                total += count
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\n{total} documents written to {OUTPUT}")


if __name__ == "__main__":
    main()
