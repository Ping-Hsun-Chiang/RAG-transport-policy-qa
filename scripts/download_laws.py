"""
Download Taiwan transport regulation PDFs from the official
Laws & Regulations Database of the R.O.C. (全國法規資料庫, law.moj.gov.tw).

Usage:
    python scripts/download_laws.py
    python scripts/download_laws.py --output-dir data/raw --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Official PDF export endpoint of law.moj.gov.tw (md=m => full text of the current version)
PDF_URL_TEMPLATE = "https://law.moj.gov.tw/LawClass/FilesType.aspx?DataId={pcode}&md=m"
PAGE_URL_TEMPLATE = "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode={pcode}"

# PCODE -> human-readable name (also used as the output filename)
LAWS: dict[str, str] = {
    "K0040001": "公路法",
    "K0040003": "汽車運輸業管理規則",
    "K0040012": "道路交通管理處罰條例",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"


def build_session() -> requests.Session:
    """HTTP session with automatic retries for transient network errors."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "Mozilla/5.0 (transport-policy-qa downloader)"})
    return session


def download_law(session: requests.Session, pcode: str, name: str, out_dir: Path, force: bool) -> bool:
    """Download one regulation PDF. Returns True on success."""
    target = out_dir / f"{pcode}_{name}.pdf"

    if target.exists() and not force:
        print(f"[SKIP] {target.name} already exists (use --force to re-download)")
        return True

    url = PDF_URL_TEMPLATE.format(pcode=pcode)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[FAIL] {name} ({pcode}): {exc}")
        return False

    # The server returns HTML on errors, so verify the PDF magic bytes.
    if not resp.content.startswith(b"%PDF"):
        print(f"[FAIL] {name} ({pcode}): response is not a PDF "
              f"(Content-Type: {resp.headers.get('Content-Type')})")
        return False

    target.write_bytes(resp.content)
    print(f"[ OK ] {target.name}  ({len(resp.content) / 1024:.1f} KB)  source: {PAGE_URL_TEMPLATE.format(pcode=pcode)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Taiwan transport regulation PDFs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Destination folder (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file exists")
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving PDFs to: {out_dir.resolve()}\n")

    session = build_session()
    results = []
    for pcode, name in LAWS.items():
        results.append(download_law(session, pcode, name, out_dir, args.force))
        time.sleep(1)  # be polite to the government server

    ok = sum(results)
    print(f"\nDone: {ok}/{len(LAWS)} succeeded.")
    return 0 if ok == len(LAWS) else 1


if __name__ == "__main__":
    sys.exit(main())
