"""
Build three decks through the Build page's chat, wait for each Forge playtest, and
check: the final deck is 60 cards, no requested card was cut, every kept swap is in
the report, and the confirmed win rate is not below the first draft's. Prints the
lists and reports for human review. Needs the stack and the sim worker running.

    python scripts/eval_playtest.py [--base http://localhost:8000]
"""

import argparse
import json
import sys
import time
import urllib.request

CASES = [
    ("Build me a mono-red aggro deck", []),
    ("I want to build a deck around Weapons Manufacturing", ["Weapons Manufacturing"]),
    ("I want a deck that can beat the current meta", []),
]
TIMEOUT_S = 15 * 60


def call(base, path, body=None):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def counts(entries):
    out = {}
    for e in entries:
        out[e["card_name"]] = out.get(e["card_name"], 0) + int(e["quantity"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8000")
    base = parser.parse_args().base + "/api"
    failures = []
    for ask, requested in CASES:
        print(f"\n=== {ask}")
        chat = call(base, "/conversations/chat", {"message": ask, "mode": "build", "format": "standard"})
        sim_id = chat.get("simulation_id")
        if not chat.get("deck"):
            failures.append(f"{ask}: no deck ({chat.get('response', '')[:120]})")
            continue
        if not sim_id:
            print("not playtested:", chat["response"][-200:])
            failures.append(f"{ask}: no playtest started")
            continue
        start = time.time()
        while True:
            run = call(base, f"/simulations/{sim_id}")
            if run["status"] in ("completed", "stopped", "failed") or time.time() - start > TIMEOUT_S:
                break
            time.sleep(15)
        print(f"status={run['status']} after {time.time() - start:.0f}s error={run.get('error')}")
        if run["status"] != "completed":
            failures.append(f"{ask}: playtest {run['status']}")
            continue
        seed, final = counts(run["deck"]["main_deck"]), counts(run["final_deck"]["main_deck"])
        report = run["report"]
        print("first draft:", "; ".join(f"{q} {n}" for n, q in seed.items()))
        print("final:      ", "; ".join(f"{q} {n}" for n, q in final.items()))
        print("overall", report["overall"], "baseline", report["baseline"], "stopped", report["stopped"])
        for c in report["changes"]:
            print(f"  -{c['copies']} {c['cut']} +{c['copies']} {c['add']}: {c['before']:.2f} -> {c['after']:.2f}")
        for e in (run["progress"] or {}).get("events", [])[-6:]:
            print("  event:", e["text"])
        checks = {
            "60 cards": sum(final.values()) == 60,
            "requested kept": all(final.get(r, 0) >= seed.get(r, 0) for r in requested),
            "changes explain the diff": {c["add"] for c in report["changes"]} >= {n for n in final if n not in seed},
            "not worse than the draft": report["baseline"] is None
                                        or report["overall"]["win_rate"] >= report["baseline"]["win_rate"],
        }
        for name, ok in checks.items():
            print(f"  {'OK ' if ok else 'FAIL'} {name}")
            if not ok:
                failures.append(f"{ask}: {name}")
    print("\nPASS" if not failures else "\nFAIL:\n" + "\n".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
