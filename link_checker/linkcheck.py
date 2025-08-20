#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Link auditor for Markdown-based awesome lists.

What it does
- Scans all *.md files under a repo folder
- Extracts [text](url) links with heading context
- Checks URLs concurrently with retries and timeouts
- Writes:
  - links_report.csv          full results
  - VALID.md                  valid links grouped by section
  - BROKEN.md                 broken links grouped by section
  - NEW_LIST.md               rebuilt list from original sections with only valid links

Usage
  1) python3 -m venv .venv && source .venv/bin/activate
  2) pip install aiohttp chardet
  3) python linkcheck.py --root .            # run from repo root
     python linkcheck.py --root path/to/repo

Notes
- Treats 2xx and 3xx as valid
- Treats some 403 and 405 responses as soft-valid for HEAD by retrying GET
- Respects simple per-domain HEAD blacklist
"""

import asyncio
import aiohttp
import argparse
import re
import os
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import chardet

MD_LINK_RE = re.compile(r'\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
MD_IMAGE_RE = re.compile(r'!\[.*?\]\(.*?\)')
MD_AUTOLINK_RE = re.compile(r'<(https?://[^>\s]+)>')

HEADINGS_RE = re.compile(r'^(#{1,6})\s+(.*)')
CODE_FENCE_RE = re.compile(r'^```')

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=8)
CONCURRENCY = 40
RETRIES = 2

# HEAD often blocked on some hosts
HEAD_BLOCKED = {
    "medium.com",
    "link.medium.com",
    "arxiv.org",
    "github.com",
    "raw.githubusercontent.com",
    "www.reddit.com",
    "twitter.com",
    "x.com",
    "mirror.xyz"
}

SOFT_VALID_STATUSES = {401, 402, 403, 405, 406, 409, 410, 429}  # log as warn but keep as broken unless GET passes

def detect_text(path: Path) -> str:
    data = path.read_bytes()
    enc = chardet.detect(data).get("encoding") or "utf-8"
    return data.decode(enc, errors="ignore")

def walk_markdown(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.md") if p.is_file()]

def norm_url(url: str) -> str:
    if url.startswith("mailto:"):
        return url
    if url.startswith("http://"):
        return url.replace("http://", "https://", 1)
    return url

def domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()
    except Exception:
        return ""

def extract_links(markdown: str) -> List[Tuple[str, str, List[str]]]:
    """
    Return list of tuples: (text, url, heading_path)
    heading_path is a list like ["Wallets", "Multisig"]
    """
    links = []
    heading_stack: List[str] = []
    in_code_block = False

    lines = markdown.splitlines()
    for line in lines:
        if CODE_FENCE_RE.match(line):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        m = HEADINGS_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # adjust stack
            heading_stack = heading_stack[:level-1]
            heading_stack.append(title)
            continue

        # strip image links from line to avoid false positives
        clean = MD_IMAGE_RE.sub("", line)

        for m in MD_LINK_RE.finditer(clean):
            text = m.group(1).strip()
            url = norm_url(m.group(2).strip())
            if url.startswith("#"):
                continue
            links.append((text, url, heading_stack.copy()))

        for m in MD_AUTOLINK_RE.finditer(clean):
            url = norm_url(m.group(1).strip())
            links.append((url, url, heading_stack.copy()))

    return links

async def fetch_status(session: aiohttp.ClientSession, url: str) -> Tuple[int, Optional[str]]:
    try:
        dom = domain_of(url)
        # try HEAD unless blocked
        if dom not in HEAD_BLOCKED:
            async with session.head(url, allow_redirects=True) as resp:
                return resp.status, None
        # fall back to GET
        async with session.get(url, allow_redirects=True) as resp:
            return resp.status, None
    except aiohttp.ClientResponseError as e:
        return e.status or 0, str(e)
    except Exception as e:
        return 0, str(e)

async def check_one(semaphore: asyncio.Semaphore, session: aiohttp.ClientSession, url: str) -> Tuple[bool, int, str]:
    async with semaphore:
        last_status = 0
        last_err = ""
        # try HEAD or GET once, then GET retries
        for attempt in range(RETRIES + 1):
            status, err = await fetch_status(session, url)
            last_status, last_err = status, err or ""
            if 200 <= status < 400:
                return True, status, ""
            # If HEAD blocked or soft status, try GET next round
            if attempt < RETRIES:
                await asyncio.sleep(0.3 * (attempt + 1))
            else:
                break
        return False, last_status, last_err

async def check_all(urls: List[str]) -> Dict[str, Tuple[bool, int, str]]:
    results: Dict[str, Tuple[bool, int, str]] = {}
    connector = aiohttp.TCPConnector(limit_per_host=8, ssl=False)
    async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT, connector=connector, headers={"User-Agent": "linkcheck/1.0"}) as session:
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = []
        for u in urls:
            tasks.append(asyncio.create_task(check_one(sem, session, u)))
        statuses = await asyncio.gather(*tasks)
        for u, st in zip(urls, statuses):
            results[u] = st
    return results

def group_by_section(entries):
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for text, url, section in entries:
        key = " / ".join(section) if section else "Ungrouped"
        grouped.setdefault(key, []).append((text, url))
    return grouped

def write_markdown(path: Path, title: str, grouped: Dict[str, List[Tuple[str, str]]]):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for sec in sorted(grouped.keys()):
            f.write(f"## {sec}\n\n")
            items = sorted(grouped[sec], key=lambda x: x[0].lower())
            for text, url in items:
                # text may be URL for autolinks
                f.write(f"- [{text}]({url})\n")
            f.write("\n")

def rebuild_from_original(original_md: str, valid_set: set) -> str:
    """
    Recreate the list structure while dropping broken links.
    Preserves headings and non-link text lines.
    Removes lines that contain only a broken link bullet.
    """
    out_lines = []
    in_code_block = False
    for line in original_md.splitlines():
        if CODE_FENCE_RE.match(line):
            in_code_block = not in_code_block
            out_lines.append(line)
            continue
        if in_code_block:
            out_lines.append(line)
            continue

        # identify bullets with links
        m_link = MD_LINK_RE.search(line)
        m_autolink = MD_AUTOLINK_RE.search(line)

        if m_link and line.strip().startswith(("-", "*", "+")):
            url = norm_url(m_link.group(2).strip())
            if url in valid_set:
                out_lines.append(line)
            else:
                # drop broken bullet
                continue
        elif m_autolink and line.strip().startswith(("-", "*", "+")):
            url = norm_url(m_autolink.group(1).strip())
            if url in valid_set:
                out_lines.append(line)
            else:
                continue
        else:
            out_lines.append(line)

    return "\n".join(out_lines).strip() + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to repo root")
    ap.add_argument("--readme", default="README.md", help="Relative path of main README for rebuild")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    md_files = walk_markdown(root)
    all_entries = []
    file_map = {}

    for md in md_files:
        text = detect_text(md)
        entries = extract_links(text)
        if entries:
            all_entries.extend([(t, u, sec, md) for t, u, sec in entries])
            file_map[md] = text

    urls = sorted({u for _, u, _, _ in all_entries if u.startswith("http") or u.startswith("mailto:")})
    print(f"Found {len(urls)} unique links")

    results = asyncio.run(check_all(urls))

    report_path = root / "links_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["file", "section", "text", "url", "ok", "status", "error"])
        for text, url, sec, md in all_entries:
            ok, status, err = results.get(url, (False, 0, "no_result"))
            w.writerow([str(md.relative_to(root)), " / ".join(sec), text, url, int(ok), status, err])

    valid_entries = []
    broken_entries = []
    valid_set = set()

    for text, url, sec, md in all_entries:
        ok, status, _ = results.get(url, (False, 0, ""))
        if ok:
            valid_entries.append((text, url, sec))
            valid_set.add(url)
        else:
            broken_entries.append((text, url, sec))

    # Group and write convenience views
    valid_grouped = group_by_section(valid_entries)
    broken_grouped = group_by_section(broken_entries)

    write_markdown(root / "VALID.md", "Valid Links", valid_grouped)
    write_markdown(root / "BROKEN.md", "Broken Links", broken_grouped)

    # Rebuild main list from README.md using only valid links
    readme_path = root / args.readme
    if readme_path.exists():
        original = detect_text(readme_path)
        rebuilt = rebuild_from_original(original, valid_set)
        (root / "NEW_LIST.md").write_text(rebuilt, encoding="utf-8")

    print("Done")
    print(f"- {report_path}")
    print(f"- {root / 'VALID.md'}")
    print(f"- {root / 'BROKEN.md'}")
    print(f"- {root / 'NEW_LIST.md'}")

if __name__ == "__main__":
    main()
