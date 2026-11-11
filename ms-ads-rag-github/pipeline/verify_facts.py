"""
Sanity check after Step 2: is the answer to each eval question somewhere in the processed data?
If a check fails, the bot can't answer that question, however good the retrieval is.
Fix the crawl scope or the cleaner before moving on to chunking.

Usage:
    python -m pipeline.verify_facts
"""
import json
import re
from pathlib import Path

PAGES = Path("data/processed/pages.jsonl")

# (eval question, regex that must appear in some section). Patterns are loose on purpose:
# the sample answers are flagged as possibly outdated, so match the concept, not the exact number.
CHECKS = [
    ("Tuition cost",                      r"\$\s?\d{1,3},\d{3}.{0,40}(per course|course)"),
    ("Scholarships",                      r"Data Science Institute Scholarship"),
    ("TOEFL / IELTS minimums",            r"TOEFL.{0,120}\d{2,3}"),
    ("IELTS score",                       r"IELTS.{0,60}\b[6-8](\.\d)?\b"),
    ("Application fee waiver",            r"fee waiver"),
    ("In-person deadlines",               r"(Priority|Final) Application Deadline"),
    ("Decision timeline",                 r"decision.{0,80}(month|rolling)"),
    ("Advising appointment link",         r"apply-psd\.uchicago\.edu"),
    ("Transcript mailing address",        r"455 N\.? Cityfront Plaza"),
    ("Online program visa sponsorship",   r"(visa).{0,150}(in-person|full-time)|(in-person|full-time).{0,150}visa"),
    ("MBA/MS application",                r"Booth.{0,200}(application|apply)"),
    ("STEM/OPT eligibility",              r"STEM.{0,40}OPT|OPT.{0,40}STEM"),
    ("12 courses to graduate",            r"\b12 courses\b|\btwelve courses\b"),
]


def main() -> None:
    pages = [json.loads(l) for l in PAGES.read_text(encoding="utf-8").splitlines() if l.strip()]
    sections = [(p["url"], s) for p in pages for s in p["sections"]]
    passed = 0
    for name, pattern in CHECKS:
        rx = re.compile(pattern, re.I | re.S)
        hits = [(url, s) for url, s in sections if rx.search(s["heading_path"] + "\n" + s["text"])]
        if hits:
            passed += 1
            url, s = hits[0]
            snippet = rx.search(s["heading_path"] + "\n" + s["text"]).group(0)[:90].replace("\n", " ")
            print(f"PASS  {name:34s} {len(hits):>2} hit(s)  e.g. '{snippet}'  <- {url}")
        else:
            print(f"FAIL  {name:34s} not found in any section")
    print(f"\n{passed}/{len(CHECKS)} eval facts present in processed data.")


if __name__ == "__main__":
    main()
