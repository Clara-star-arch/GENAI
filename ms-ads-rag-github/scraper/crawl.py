"""
Step 1: BFS crawler for the MS-ADS site.

Saves raw HTML to data/raw/<hash>.html and one manifest row per page to
data/raw/manifest.jsonl. Raw HTML is kept so cleaning can be re-run without
re-crawling.

Usage:
    python -m scraper.crawl                 # normal crawl
    python -m scraper.crawl --playwright    # re-fetch thin (JS-rendered) pages with a browser
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from scraper import config as C

RAW_DIR = Path("data/raw")
MANIFEST = RAW_DIR / "manifest.jsonl"


# ---------- URL helpers ----------

def normalize(url: str) -> str:
    """Drop fragments and query strings, force https, ensure trailing slash on paths."""
    p = urlparse(url.strip())
    # Drop stray encoded whitespace segments, e.g. ".../online-program/%20/" from an href with a trailing space.
    path = re.sub(r"(/(%20|\s)+)+(?=/|$)", "", p.path) or "/"
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    return urlunparse(("https", p.netloc.lower(), path, "", "", ""))


def is_excluded(url: str) -> bool:
    low = url.lower()
    return low.endswith(C.EXCLUDE_EXTENSIONS) or any(x in low for x in C.EXCLUDE_PATTERNS)


def scope_of(url: str) -> str | None:
    """'follow' = crawl and explore links, 'leaf' = crawl only, None = skip."""
    if is_excluded(url):
        return None
    if any(url.startswith(p) for p in C.IN_SCOPE_PREFIXES):
        return "follow"
    if any(url.startswith(p) for p in C.LEAF_PREFIXES):
        return "leaf"
    return None


def url_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


# ---------- robots.txt ----------

_robots: dict[str, robotparser.RobotFileParser] = {}

def allowed(url: str, session: requests.Session) -> bool:
    host = urlparse(url).netloc
    if host not in _robots:
        rp = robotparser.RobotFileParser()
        try:
            r = session.get(f"https://{host}/robots.txt", timeout=C.TIMEOUT_SEC)
            rp.parse(r.text.splitlines() if r.ok else [])
        except requests.RequestException:
            rp.parse([])
        _robots[host] = rp
    return _robots[host].can_fetch(C.USER_AGENT, url)


# ---------- fetching ----------

def visible_text_len(html: str) -> int:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        t.decompose()
    return len(soup.get_text(" ", strip=True))


def fetch_requests(url: str, session: requests.Session) -> tuple[int, str, str]:
    r = session.get(url, timeout=C.TIMEOUT_SEC, allow_redirects=True)
    ctype = r.headers.get("content-type", "")
    if "html" not in ctype:
        return r.status_code, "", r.url
    return r.status_code, r.text, r.url


def fetch_playwright(url: str) -> str:
    from playwright.sync_api import sync_playwright  # optional dependency
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(user_agent=C.USER_AGENT)
        page.goto(url, wait_until="networkidle", timeout=C.TIMEOUT_SEC * 1000)
        html = page.content()
        browser.close()
    return html


def extract_links(html: str, base: str) -> list[str]:
    """Links from the WHOLE page, including sidebar menus, which are the best map of the site."""
    soup = BeautifulSoup(html, "lxml")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        out.append(normalize(urljoin(base, href)))
    return out


# ---------- main loop ----------

def crawl(use_playwright: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": C.USER_AGENT})

    queue: deque[tuple[str, int, str | None]] = deque((normalize(u), 0, None) for u in C.SEEDS)
    seen: set[str] = {u for u, _, _ in queue}
    rows: list[dict] = []

    while queue and len(rows) < C.MAX_PAGES:
        url, depth, parent = queue.popleft()
        scope = scope_of(url) or ("follow" if depth == 0 else None)
        if scope is None:
            continue
        if not allowed(url, session):
            print(f"[robots] skip {url}")
            continue

        try:
            status, html, final_url = fetch_requests(url, session)
        except requests.RequestException as e:
            print(f"[error] {url}: {e}")
            rows.append({"url": url, "status": None, "error": str(e), "depth": depth})
            continue

        thin = bool(html) and visible_text_len(html) < C.MIN_TEXT_CHARS
        rendered_with = "requests"
        if thin and use_playwright:
            try:
                html = fetch_playwright(url)
                rendered_with = "playwright"
                thin = visible_text_len(html) < C.MIN_TEXT_CHARS
            except Exception as e:  # noqa: BLE001
                print(f"[playwright error] {url}: {e}")

        row = {
            "url": url,
            "final_url": normalize(final_url),
            "status": status,
            "depth": depth,
            "parent": parent,
            "scope": scope,
            "thin": thin,
            "rendered_with": rendered_with,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        if status == 200 and html:
            fid = url_id(url)
            (RAW_DIR / f"{fid}.html").write_text(html, encoding="utf-8")
            row["file"] = f"{fid}.html"
            title = BeautifulSoup(html, "lxml").title
            row["title"] = title.get_text(strip=True) if title else ""

            if scope == "follow" and depth < C.MAX_DEPTH:
                for link in extract_links(html, final_url):
                    if link not in seen and scope_of(link):
                        seen.add(link)
                        queue.append((link, depth + 1, url))

        rows.append(row)
        flag = " THIN" if thin else ""
        print(f"[{status}] d={depth} {scope:6s} {url}{flag}")
        time.sleep(C.REQUEST_DELAY_SEC)

    with MANIFEST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ok = sum(1 for r in rows if r.get("status") == 200)
    thin = [r["url"] for r in rows if r.get("thin")]
    print(f"\nSaved {ok} pages to {RAW_DIR}/ ({len(rows)} attempted).")
    if thin:
        print(f"{len(thin)} thin pages (maybe JS-rendered); re-run with --playwright:")
        for u in thin:
            print("  ", u)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--playwright", action="store_true", help="re-fetch thin pages with a headless browser")
    crawl(ap.parse_args().playwright)
