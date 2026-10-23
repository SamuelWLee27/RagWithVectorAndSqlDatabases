import sys
import time
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "docs" / "sources.yaml"
RAW_DIR = ROOT / "docs" / "raw"

HEADERS = {"User-Agent": "Mozilla/5.0 (personal RAG learning project)"}
DELAY_SECONDS = 1.0

def extension_for(content_type: str) -> str | None:
    if "pdf" in content_type:
        return ".pdf"
    if "html" in content_type:
        return ".html"
    return None


def download(entry: dict, client: httpx.Client, force: bool) -> str:
    doc_id, url = entry["id"], entry["url"]
    existing = list(RAW_DIR.glob(f"{doc_id}.*"))
    if existing and not force:
        return f"skip      {existing[0].name}"

    resp = client.get(url)
    resp.raise_for_status()

    ext = extension_for(resp.headers.get("content-type", ""))
    if ext is None and resp.content[:5] == b"%PDF-":
        ext = ".pdf"
    if ext is None:
        return f"FAILED    {doc_id}: unexpected content-type {resp.headers.get('content-type')}"

    path = RAW_DIR / f"{doc_id}{ext}"
    path.write_bytes(resp.content)
    return f"saved     {path.name} ({len(resp.content) / 1024:.0f} KB)"


def main() -> None:
    force = "--force" in sys.argv
    entries = yaml.safe_load(SOURCES.read_text()) or []
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ids = [e["id"] for e in entries]
    if len(ids) != len(set(ids)):
        sys.exit("Duplicate ids in sources.yaml")

    failures = 0
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        for entry in entries:
            try:
                msg = download(entry, client, force)
            except httpx.HTTPError as e:
                msg = f"FAILED    {entry['id']}: {e}"
            if msg.startswith("FAILED"):
                failures += 1
            print(msg)
            if not msg.startswith("skip"):
                time.sleep(DELAY_SECONDS)

    print(f"\n{len(entries) - failures}/{len(entries)} documents ready in {RAW_DIR}")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()