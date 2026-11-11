"""
Terminal chat for quick testing before the UI.

Usage:
    python -m rag.chat            # interactive; type 'exit' to quit, 'reset' to clear history
    python -m rag.chat -q "What is the tuition?"     # one question
"""
import argparse

from rag.chain import RAGChain


def show(r) -> None:
    print(f"\n{r.answer}\n")
    for n in r.cited:
        h = r.sources[n - 1]
        print(f"  [{n}] {h.metadata['heading_path'][:80]}\n      {h.metadata['url']}")
    active = {k: v for k, v in r.flags.items() if v not in (False, None, [], {})}
    if r.search_query and r.search_query != r.question:
        print(f"  (searched for: {r.search_query})")
    print(f"  flags: {active or 'none'}   time: {r.timings.get('total_s')}s\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question")
    args = ap.parse_args()
    bot = RAGChain()
    if args.question:
        show(bot.answer(args.question))
        return
    history: list[tuple[str, str]] = []
    print("MS-ADS Assistant. Type 'exit' to quit, 'reset' to clear the conversation.")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "reset":
            history = []
            print("(conversation cleared)")
            continue
        if q:
            r = bot.answer(q, history)
            show(r)
            history.append((r.question, r.answer))


if __name__ == "__main__":
    main()
