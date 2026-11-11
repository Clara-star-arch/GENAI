"""
Step 2: turn raw HTML into structured, tagged sections.

Input:  data/raw/manifest.jsonl + data/raw/*.html   (from scraper/crawl.py)
Output: data/processed/pages.jsonl                    one JSON object per page
        data/processed/preview.md                     human-readable dump for spot checks

Each page:
{
  "url", "title", "category", "program_type",
  "sections": [{"heading", "heading_path", "level", "program_type", "text"}]
}

Usage:
    python -m pipeline.clean
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# ---------- 1. Strip site chrome ----------

# Candidate containers for the main content, tried in order.
MAIN_SELECTORS = [
    "main", "[role=main]", "article", "#main", "#content", ".entry-content",
    ".page-content", ".site-main",
]

# Removed wherever they appear inside the main container.
DROP_TAGS = ["script", "style", "noscript", "iframe", "svg", "form",
             "header", "footer", "nav", "aside"]
# (?<![a-z]) stops "search" from matching "research", "menu" from matching inside words, etc.
DROP_CLASS_RE = re.compile(
    r"(?<![a-z])(sidebar|side-nav|sidenav|menu|breadcrumb|share|social|cookie|newsletter|"
    r"skip-link|screen-reader|site-header|site-footer|related|also-of-interest|search)",
    re.I,
)


def find_main(soup: BeautifulSoup) -> Tag:
    for sel in MAIN_SELECTORS:
        node = soup.select_one(sel)
        if node and len(node.get_text(strip=True)) > 200:
            return node
    return soup.body or soup


def strip_chrome(main: Tag) -> None:
    for sel in DROP_TAGS:
        for t in main.select(sel):
            t.decompose()
    for t in main.find_all(True):
        if getattr(t, "decomposed", False) or t.attrs is None:
            continue
        attrs = " ".join(t.get("class", [])) + " " + (t.get("id") or "")
        # Keep accordion containers even if their class mentions "menu"-like words.
        if DROP_CLASS_RE.search(attrs) and "accordion" not in attrs.lower():
            t.decompose()


# ---------- 2. Walk content into sections ----------

HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}
# Tags that make an element a structural container rather than a single line of text.
STRUCTURAL = ["p", "div", "table", "ul", "ol", "dl", "section", "article", "blockquote",
              "h1", "h2", "h3", "h4", "h5", "h6", "pre", "figure"]
BLOCK_TEXT_TAGS = {"p", "li", "dd", "dt", "blockquote", "figcaption", "address", "pre"}

# Accordion / FAQ question markers -> treated as level-5 headings so that
# each question starts its own section and keeps its answer with it.
ACCORDION_TITLE_RE = re.compile(r"accordion[-_]?(title|header|heading|toggle|button|trigger)|faq[-_]?(question|title)", re.I)


def clean_ws(s: str) -> str:
    s = s.replace("\xa0", " ").replace("​", "")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    s = re.sub(r" +([.,;:!?)])", r"\1", s)   # "policy (url) ." -> "policy (url)."
    s = re.sub(r"(\d),,+(\d)", r"\1,\2", s)   # "$81,,960" (split <sup>/<span> markup) -> "$81,960"
    return s.strip()


def cell_text(cell: Tag) -> str:
    return clean_ws(cell.get_text(" ", strip=True)).replace("|", "/").replace("\n", " ")


def table_to_md(table: Tag) -> str:
    rows = []
    for tr in table.find_all("tr"):
        cells = [cell_text(c) for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)


def is_accordion_title(t: Tag) -> bool:
    if t.name == "summary":
        return True
    if t.name == "button" and t.get("aria-controls"):
        return True
    cls = " ".join(t.get("class", []))
    return bool(ACCORDION_TITLE_RE.search(cls))


ACCORDION_ATTR = "data-accordion-q"


def attach_accordion_titles(main: Tag, soup: BeautifulSoup) -> int:
    """Move every accordion/tab question to the top of the panel it opens, as a heading.

    Handles the layouts seen on WordPress sites:
      - adjacent:  <button aria-controls="p1">Q</button><div id="p1">A</div>
      - tab list:  <ul><li><a href="#accordion-tab-0">Q</a>...</ul> ... <div id="accordion-tab-0">A</div>
      - repeated ids per category (each FAQ group restarts at accordion-tab-0): the panel is
        searched for in the trigger's nearest enclosing container first, so the right one is found.
      - nested (category tab -> question accordion): the inner question gets a deeper heading level.
    Returns the number of questions attached."""
    triggers = main.select("[aria-controls], a[href^='#'], [data-target^='#'], [data-bs-target^='#']")
    attached = 0
    for trig in triggers:
        if getattr(trig, "decomposed", False) or trig.attrs is None:
            continue
        target = (trig.get("aria-controls") or trig.get("data-bs-target") or
                  trig.get("data-target") or trig.get("href") or "").lstrip("#").strip()
        title = clean_ws(trig.get_text(" ", strip=True))
        if not target or not title or len(title) > 300:
            continue
        panel = None
        for anc in trig.parents:                      # nearest scope first
            panel = anc.find(id=target)
            if panel is not None or anc is main:
                break
        if panel is None or panel is trig or trig in panel.descendants or panel in trig.parents:
            continue                                   # skip-links like "#main", self-references
        if panel.has_attr("data-accordion-panel"):
            continue    # first trigger wins: later in-text links to the same panel stay as links
        panel["data-accordion-panel"] = "1"
        depth = sum(1 for a in panel.parents if a.has_attr("data-accordion-panel"))
        h = soup.new_tag(f"h{min(6, 5 + depth)}")
        h[ACCORDION_ATTR] = "1"
        h.string = title
        panel.insert(0, h)
        # Remove the trigger; if it sat alone in a heading or list item, remove that wrapper too.
        wrapper = trig.parent
        trig.decompose()
        if wrapper is not None and wrapper.name in ("li", "h2", "h3", "h4", "h5", "h6", "dt") \
                and not wrapper.get_text(strip=True):
            wrapper.decompose()
        attached += 1
    return attached


class SectionBuilder:
    def __init__(self, page_title: str):
        self.sections: list[dict] = []
        self.stack: list[tuple[int, str]] = []  # (level, heading)
        self._start(page_title or "Overview", 1)

    def _start(self, heading: str, level: int, accordion: bool = False) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, heading))
        self.sections.append({
            "heading": heading,
            "heading_path": " > ".join(h for _, h in self.stack),
            "level": level,
            "accordion": accordion,
            "parts": [],
        })

    def heading(self, text: str, level: int, accordion: bool = False) -> None:
        text = clean_ws(text)
        if not text:
            return
        cur = self.sections[-1]
        # Merge consecutive headings with no body (e.g. H1 page title then H1 repeated).
        if not cur["parts"] and cur["heading"].lower() == text.lower():
            return
        self._start(text, level, accordion)

    def text(self, s: str) -> None:
        indent = s[: len(s) - len(s.lstrip(" "))]   # keep nested-list indentation
        s = clean_ws(s)
        if s:
            s = indent + s
            self.sections[-1]["parts"].append(s)

    def result(self) -> list[dict]:
        out = []
        for s in self.sections:
            body = "\n".join(s.pop("parts"))
            if body:
                out.append({**s, "text": body})
        return out


def walk(node: Tag, sb: SectionBuilder, list_depth: int = 0) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            if child.parent and child.parent.name not in BLOCK_TEXT_TAGS:
                txt = str(child).strip()
                if len(txt) > 30:          # stray text directly inside a div
                    sb.text(txt)
            continue
        if not isinstance(child, Tag):
            continue

        name = child.name
        if name in HEADING_TAGS:
            sb.heading(child.get_text(" ", strip=True), HEADING_TAGS[name], child.has_attr(ACCORDION_ATTR))
        elif is_accordion_title(child):
            # Trigger without a panel id, e.g. an FAQ category tab (<a href="#">Online Program</a>).
            # Level 4 so the questions attached inside it (level 5+) nest under it.
            sb.heading(child.get_text(" ", strip=True), 4, True)
        elif name == "table":
            sb.text(table_to_md(child))
        elif name == "li" and child.find([t for t in STRUCTURAL if t not in ("ul", "ol")]):
            # A list item that holds headings/paragraphs/panels (e.g. FAQ categories built as
            # <li> accordions) is a container, not a bullet: walk it instead of flattening it.
            walk(child, sb, list_depth)
        elif name == "li":
            # Plain bullet. Nested lists: keep the item's own text, then recurse into sub-lists.
            sub_lists = child.find_all(["ul", "ol"], recursive=False)
            for sl in sub_lists:
                sl.extract()
            sb.text("  " * list_depth + "- " + child.get_text(" "))
            for sl in sub_lists:
                walk(sl, sb, list_depth + 1)
        elif name in BLOCK_TEXT_TAGS:
            sb.text(child.get_text(" "))
        elif name == "a" and child.get_text(strip=True):
            # A bare link outside a paragraph (e.g. a CTA). inline_links() already appended its URL.
            sb.text(child.get_text(" "))
        elif name == "br":
            continue
        elif not child.find(STRUCTURAL + ["li"]):
            # Container with only inline content, e.g. <div><i>University of Chicago</i></div>:
            # emit it as one line so short fragments are not lost.
            txt = child.get_text(" ")
            if txt.strip():
                sb.text(txt)
        else:
            walk(child, sb, list_depth)


def inline_links(main: Tag, base_url: str) -> None:
    """Append URLs to link text, e.g. 'schedule an appointment (https://...)'.
    The eval set expects answers like the advising-portal URL, so links must survive."""
    from urllib.parse import urljoin
    for a in main.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("#", "javascript:")) or not a.get_text(strip=True):
            continue
        full = urljoin(base_url, href)
        a.string = f"{a.get_text(' ', strip=True)} ({full})"


# ---------- 3. Tagging ----------

CATEGORY_RULES = [  # (category, url substrings, title/heading keywords)
    ("tuition",     ["tuition"],                       ["tuition", "scholarship", "financial aid", "fees"]),
    ("deadlines",   ["deadline", "events-deadlines"],  ["deadline"]),
    ("admissions",  ["how-to-apply", "admission"],     ["apply", "application", "admission", "requirement", "transcript"]),
    ("faq",         ["faq"],                           ["faq"]),
    ("international", ["international", "english-language"], ["international", "visa", "opt", "toefl", "ielts", "english"]),
    ("curriculum",  ["course", "curriculum", "capstone", "progression"], ["course", "curriculum", "capstone", "elective", "core"]),
    ("faculty",     ["faculty", "instructor", "people"], ["faculty", "instructor"]),
    ("careers",     ["career", "outcome"],             ["career", "outcome", "employ"]),
    ("joint_degree", ["mba", "booth", "joint"],        ["mba/ms", "booth", "joint degree"]),
    ("online",      ["online-program"],                []),
    ("in_person",   ["in-person-program"],             []),
]


def page_category(url: str, title: str) -> str:
    u, t = url.lower(), title.lower()
    for cat, url_keys, title_keys in CATEGORY_RULES:
        if any(k in u for k in url_keys):
            return cat
    for cat, _, title_keys in CATEGORY_RULES:
        if any(k in t for k in title_keys):
            return cat
    return "overview"


ONLINE_RE = re.compile(r"\bonline\b", re.I)
INPERSON_RE = re.compile(r"\bin[- ]person\b|\bfull[- ]time\b|\bon[- ]campus\b", re.I)


def program_type_from_text(text: str) -> str:
    on, ip = bool(ONLINE_RE.search(text)), bool(INPERSON_RE.search(text))
    if on and not ip:
        return "online"
    if ip and not on:
        return "in_person"
    return "both"


def page_program_type(url: str) -> str:
    u = url.lower()
    if "online-program" in u:
        return "online"
    if "in-person-program" in u:
        return "in_person"
    return "both"


def section_program_type(page_pt: str, sec: dict) -> str:
    if page_pt != "both":
        return page_pt
    # The heading is the strongest signal ("Online Program" FAQ category, "In-Person Deadlines").
    # Body text is NOT used: answers often mention both programs ("in-person decisions take 1-2
    # months, online decisions are rolling"), and a wrong tag would hide the chunk from one audience.
    return program_type_from_text(sec["heading_path"])


# ---------- 4. Cross-page boilerplate removal ----------

def remove_repeated_lines(pages: list[dict], min_share: float = 0.4, min_pages: int = 4) -> int:
    """Drop lines that appear on a large share of pages (leftover CTAs, contact blurbs)."""
    counts: Counter[str] = Counter()
    for p in pages:
        lines = {ln for s in p["sections"] for ln in s["text"].split("\n")}
        counts.update(lines)
    threshold = max(min_pages, int(len(pages) * min_share))
    boiler = {ln for ln, c in counts.items() if c >= threshold and len(ln) < 300}
    removed = 0
    for p in pages:
        for s in p["sections"]:
            kept = [ln for ln in s["text"].split("\n") if ln not in boiler]
            removed += len(s["text"].split("\n")) - len(kept)
            s["text"] = "\n".join(kept)
        p["sections"] = [s for s in p["sections"] if s["text"].strip()]
    return removed


# ---------- 5. Driver ----------

def clean_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("h1") or soup.title
    title = clean_ws(title_tag.get_text(" ", strip=True)) if title_tag else url
    title = re.split(r"\s+[|–-]\s+", title)[0]  # "FAQs | Data Science Institute" -> "FAQs"

    main = find_main(soup)
    strip_chrome(main)
    n_qa = attach_accordion_titles(main, soup)
    inline_links(main, url)
    for br in main.find_all("br"):          # keep line breaks (addresses, multi-line answers)
        br.replace_with("\n")

    sb = SectionBuilder(title)
    walk(main, sb)
    sections = sb.result()

    page_pt = page_program_type(url)
    for s in sections:
        s["program_type"] = section_program_type(page_pt, s)

    return {
        "url": url,
        "title": title,
        "category": page_category(url, title),
        "program_type": page_pt,
        "n_accordion_questions": n_qa,
        "sections": sections,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = [json.loads(l) for l in (RAW_DIR / "manifest.jsonl").read_text().splitlines() if l.strip()]

    pages, seen_final = [], set()
    for row in manifest:
        if row.get("status") != 200 or not row.get("file"):
            continue
        final = row.get("final_url") or row["url"]
        if final in seen_final:              # two URLs redirecting to the same page
            continue
        seen_final.add(final)
        html = (RAW_DIR / row["file"]).read_text(encoding="utf-8")
        page = clean_page(html, final)
        if page["sections"]:
            pages.append(page)

    removed = remove_repeated_lines(pages)
    pages = [p for p in pages if p["sections"]]

    with (OUT_DIR / "pages.jsonl").open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with (OUT_DIR / "preview.md").open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(f"\n\n# {p['title']}\n`{p['url']}` · category={p['category']} · program={p['program_type']}\n")
            for s in p["sections"]:
                f.write(f"\n## {s['heading_path']}  _[{s['program_type']}]_\n{s['text']}\n")

    n_sec = sum(len(p["sections"]) for p in pages)
    cats = Counter(p["category"] for p in pages)
    print(f"{len(pages)} pages, {n_sec} sections, {removed} boilerplate lines removed")
    print("categories:", dict(cats))
    for p in pages:
        n_acc = sum(1 for s in p["sections"] if s.get("accordion"))
        if n_acc:
            print(f"  {n_acc:>3} accordion Q&A sections  <- {p['url']}")
    print(f"wrote {OUT_DIR/'pages.jsonl'} and {OUT_DIR/'preview.md'}")


if __name__ == "__main__":
    main()
