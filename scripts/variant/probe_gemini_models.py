#!/usr/bin/env python3
"""Which Gemini models does THIS key actually have? (read-only)

WHY THIS EXISTS
---------------
configs/models.yaml pins gemini-3.5-flash and gemini-3.5-flash-lite because
those are what the key was verified against on 2026-07-27. Adding a Pro row is
not a matter of typing a model id into the config and hoping: a wrong id fails
at the first API call, and a right id can still be unusable because the Google
AI Studio FREE tier meters each model separately (flash was measured at 20
requests/day, flash-lite at >260 -- see the notes in models.yaml). So the two
questions "does my key see Pro at all" and "can it afford a sweep" are
different, and neither can be answered by guessing an id.

WHAT THIS DOES / DOES NOT DO
----------------------------
Does:      one models.list() call -- free metadata, no generation.
Does NOT:  generate anything, so it consumes NO per-model generation quota.
Does NOT:  print, log, or write the key anywhere. It is read from the
           environment and handed straight to the client.

Pass --try-generate to additionally send ONE tiny real prompt to a chosen
model. That DOES spend one request of that model's daily quota, which is why
it is opt-in rather than the default: on a 20/day model, five careless probes
are a quarter of the day's budget.

USAGE
-----
    set -a; source .env; set +a          # loads GEMINI_API_KEY into the env
    python scripts/variant/probe_gemini_models.py
    python scripts/variant/probe_gemini_models.py --try-generate gemini-3.5-pro

Needs:  pip install -e ".[gemini]"
"""
import argparse, os, sys

# Scripts live one level deeper than the repo root (scripts/<block>/x.py),
# so the root is three dirnames up, not two.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--try-generate", metavar="MODEL_ID", default=None,
                    help="send ONE real prompt to this model. Spends one "
                         "request of its daily quota -- opt-in on purpose.")
    ap.add_argument("--all", action="store_true",
                    help="list every model, not just the generate-capable ones")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("GEMINI_API_KEY is not set. Load it without echoing it:\n"
              "    set -a; source .env; set +a")
        return 1
    # Fingerprint only -- enough to tell two keys apart in a transcript,
    # useless to anyone who reads it.
    print(f"key: {len(key)} chars, ends {key[-4:]!r}\n")

    try:
        from google import genai
    except ImportError:
        print('google-genai not installed:  pip install -e ".[gemini]"')
        return 1
    client = genai.Client(api_key=key)

    try:
        models = list(client.models.list())
    except Exception as e:                                      # noqa: BLE001
        print(f"models.list() failed: {e}")
        return 1

    rows = []
    for m in models:
        name = (getattr(m, "name", "") or "").replace("models/", "")
        actions = list(getattr(m, "supported_actions", None) or [])
        if not args.all and actions and "generateContent" not in actions:
            continue
        rows.append((name, getattr(m, "input_token_limit", None),
                     getattr(m, "output_token_limit", None)))

    print(f"=== {len(rows)} model(s) visible to this key ===")
    pro = [r for r in rows if "pro" in r[0].lower()]
    for name, tin, tout in sorted(rows):
        mark = "  <-- PRO" if "pro" in name.lower() else ""
        print(f"  {name:<44} in={tin!s:>9} out={tout!s:>7}{mark}")

    print()
    if pro:
        print(f"PRO AVAILABLE: {[p[0] for p in pro]}")
        print("Visible != affordable. The free tier meters each model "
              "separately and Pro tiers are usually capped tighter than "
              "flash, so confirm the daily cap before adding a Pro row:")
        print(f"    python {os.path.relpath(__file__)} "
              f"--try-generate {pro[0][0]}")
    else:
        print("NO Pro model is visible to this key. Either the project has no "
              "Pro access, or Pro is not offered on its tier -- adding a Pro "
              "row to models.yaml would fail at the first call.")

    if args.try_generate:
        print(f"\n=== one real call to {args.try_generate} "
              f"(spends 1 request of its daily quota) ===")
        from google.genai import types
        try:
            r = client.models.generate_content(
                model=args.try_generate,
                contents="Reply with exactly one word: PM1",
                config=types.GenerateContentConfig(
                    system_instruction="Reply with only the option name.",
                    max_output_tokens=32, temperature=0))
            reply = (getattr(r, "text", "") or "").strip()[:80]
            print(f"OK  reply={reply!r}")
            print("    -> this model id works with GeminiAgent as-is; add a row "
                  "to configs/models.yaml. Give it a cache_path if its daily "
                  "cap is low, so a sweep is resumable across days.")
        except Exception as e:                                  # noqa: BLE001
            msg = str(e)
            print(f"FAIL  {msg[:400]}")
            if "quota" in msg.lower() or "429" in msg:
                print("    -> quota, not access. The id is right; the tier "
                      "cannot afford it today. Same situation models.yaml "
                      "already documents for gemini-3.5-flash (20/day).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
