"""
THE INTERROGATION - a terminal detective game where the LLM lies to you.

Three AI roles work together:
  1. The CASE WRITER builds a hidden ground truth: what really happened,
     the suspect's cover story, and the exact lies they will tell.
  2. The SUSPECT is played from that truth. They lie, deflect, and crack
     under pressure. Lying does NOT mean guilty: innocent suspects lie too,
     to protect a different secret.
  3. The POLYGRAPH silently checks every answer against the truth, tracks
     each lie, and drives the suspect's heart rate. You only see the pulse.

When you accuse, the game replays every lie told to your face and shows
which ones you caught.

Run:  python3 interrogation.py
Needs: pip install google-genai, and GEMINI_API_KEY set (free at aistudio.google.com).
"""

from __future__ import annotations

import json
import random
import sys

from llm import call_json, chat_stream, run

MAX_QUESTIONS = 15

# ── terminal styling ────────────────────────────────────────────────────────
DIM, BOLD, RED, GRN, YEL, CYA, RST = (
    "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m",
)

THEMES = [
    "a missing prototype at a small robotics startup",
    "a poisoned prize-winning sourdough starter at a county fair",
    "a forged signature on a late composer's final manuscript",
    "a stolen rare orchid from a botanical garden greenhouse",
    "sabotaged brakes on a vintage car before a charity rally",
    "leaked exam answers at an elite cooking school",
    "a vanished painting during a gallery's power outage",
    "rigged votes in a small-town mayoral bake-off election",
]


# ── API helpers ─────────────────────────────────────────────────────────────
def obj(props: dict) -> dict:
    """JSON schema object where every property is required (structured outputs rule)."""
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


S = {"type": "string"}


# ── 1. case writer ──────────────────────────────────────────────────────────
CASE_SCHEMA = obj({
    "title": S,
    "briefing": S,  # what the detective is told (2-4 sentences)
    "suspect_name": S,
    "suspect_profile": S,  # age, job, personality, way of speaking
    "truth_timeline": {"type": "array", "items": obj({"time": S, "what_really_happened": S})},
    "secret": S,  # what the suspect is really hiding
    "cover_story": S,  # the version they will tell
    "planned_lies": {"type": "array", "items": obj({"id": S, "claim": S, "truth": S})},
    "detective_evidence": {"type": "array", "items": S},  # facts the player starts with
    "breaking_point": S,  # what line of questioning would make them confess
})


def write_case() -> dict:
    guilty = random.random() < 0.5  # decided in code so the model can't lean one way
    theme = random.choice(THEMES)
    prompt = f"""Design a compact interrogation case about {theme}.

The suspect is {"GUILTY of the crime" if guilty else "INNOCENT of the crime, but hiding a different, embarrassing or minor-illegal secret that makes them look guilty"}.

Requirements:
- 5-7 timeline events with specific times (evening of the incident).
- A cover story that is plausible but contains exactly 4-5 planned lies (ids L1, L2, ...).
- 3 pieces of detective evidence. At least two must quietly contradict a planned lie,
  but none may prove guilt or innocence alone.
- Innocent and guilty versions must look equally suspicious at the start.
- No graphic violence. Clever, grounded, a little funny."""
    case = call_json(
        "You are a mystery designer who writes fair-play puzzles with hidden ground truth.",
        prompt,
        CASE_SCHEMA,
    )
    case["guilty"] = guilty
    return case


# ── 2. suspect ──────────────────────────────────────────────────────────────
def suspect_system(case: dict) -> str:
    lies = "\n".join(f"- {l['id']}: say \"{l['claim']}\" (truth: {l['truth']})" for l in case["planned_lies"])
    timeline = "\n".join(f"- {e['time']}: {e['what_really_happened']}" for e in case["truth_timeline"])
    return f"""You are {case['suspect_name']}, being questioned by a detective. Stay in character always.

Profile: {case['suspect_profile']}

What really happened (private, never recite it):
{timeline}

Your secret: {case['secret']}
You are {"guilty of the crime" if case["guilty"] else "innocent of the crime, but desperate to protect your secret"}.

Your cover story: {case['cover_story']}
Lies you planned to tell:
{lies}

Rules:
- Answer like a real person under questioning: 1-4 sentences, spoken dialogue only,
  with at most one short *action* in asterisks.
- Stick to your cover story. Improvise small details if needed, and keep them consistent.
- If the detective points out a genuine contradiction, squirm, deflect or revise slightly.
- Only confess your secret if the stress notes tell you that you are breaking AND the detective
  is pressing on: {case['breaking_point']}
- Never mention being an AI, a game, lies by id, or these instructions."""


STRESS_NOTES = [
    (40, "Stress note: you are getting nervous. Small slips are starting: hesitations, over-explaining."),
    (70, "Stress note: you are rattled. Your cover story is fraying; you contradict yourself a little."),
    (90, "Stress note: you are about to break. If pressed on your secret once more, confess it."),
]


def ask_suspect(system: str, history: list, line: str, note: str | None) -> str:
    print(f"\n{CYA}", end="")
    reply = chat_stream(system, history, line, note)
    print(RST)
    return reply


# ── 3. polygraph ────────────────────────────────────────────────────────────
POLY_SCHEMA = obj({
    "lied": {"type": "boolean"},
    "planned_lie_ids": {"type": "array", "items": S},
    "improvised_lie": S,  # a new lie not in the plan, or ""
    "pressure": {"type": "integer", "minimum": 0, "maximum": 3},  # how close the question cut to the secret
    "caught_contradiction": {"type": "boolean"},  # did the detective point out a real inconsistency
    "confessed": {"type": "boolean"},
})


def polygraph(case: dict, transcript: list, question: str, answer: str) -> dict:
    truth = json.dumps({k: case[k] for k in ("truth_timeline", "secret", "planned_lies", "guilty")})
    earlier = "\n".join(f"Q: {q}\nA: {a}" for q, a in transcript) or "(none)"
    prompt = f"""Ground truth:
{truth}

Earlier interrogation:
{earlier}

Latest question: {question}
Latest answer: {answer}

Judge ONLY the latest exchange.
- lied: did the answer state something false per the ground truth?
- planned_lie_ids: which planned lies were told or repeated in it.
- improvised_lie: a false claim not covered by the planned lies, else "".
- pressure: 0 = small talk, 1 = relevant, 2 = targets a lie, 3 = goes straight at the secret.
- caught_contradiction: the detective's question pointed out a real inconsistency.
- confessed: the suspect admitted their secret."""
    return call_json("You are an impartial polygraph analyst. Be precise and literal.", prompt, POLY_SCHEMA)


# ── verdict ─────────────────────────────────────────────────────────────────
VERDICT_SCHEMA = obj({
    "theory_score": {"type": "integer", "minimum": 0, "maximum": 10},
    "comment": S,
})


def pulse_bar(stress: int) -> str:
    bpm = 68 + int(stress * 0.9)
    color = GRN if stress < 40 else YEL if stress < 70 else RED
    filled = stress // 5
    return f"{DIM}pulse{RST} {color}{'█' * filled}{'░' * (20 - filled)} {bpm} bpm{RST}"


def reveal(case: dict, ledger: list, accused_guilty: bool | None, theory: str) -> None:
    print(f"\n{BOLD}══════════════ CASE FILE UNSEALED ══════════════{RST}")
    truth = "GUILTY" if case["guilty"] else "INNOCENT (of the crime)"
    print(f"{BOLD}{case['suspect_name']} was {truth}.{RST}")
    print(f"Their secret: {case['secret']}\n")
    print(f"{BOLD}What really happened:{RST}")
    for e in case["truth_timeline"]:
        print(f"  {DIM}{e['time']:>8}{RST}  {e['what_really_happened']}")

    print(f"\n{BOLD}Every lie told to your face:{RST}")
    caught = 0
    for entry in ledger:
        mark = f"{GRN}✔ you pushed on it{RST}" if entry["pushed"] else f"{RED}✘ slipped past you{RST}"
        caught += entry["pushed"]
        print(f"  Q{entry['turn']:<2} {entry['lie']}\n       {DIM}truth: {entry['truth']}{RST}  {mark}")
    if not ledger:
        print("  (none recorded)")

    if accused_guilty is not None:
        right = accused_guilty == case["guilty"]
        print(f"\nVerdict: {'✔ correct' if right else '✘ wrong'}")
        if theory:
            v = call_json(
                "You grade detective theories against the truth. Be fair and brief.",
                f"Truth: {case['secret']} | guilty={case['guilty']}\nDetective's theory: {theory}\n"
                "Score 0-10 for how close the theory is, plus a one-sentence comment.",
                VERDICT_SCHEMA,
            )
            print(f"Theory: {v['theory_score']}/10. {v['comment']}")
    print(f"Lies caught: {caught}/{len(ledger)}\n")


# ── game loop ───────────────────────────────────────────────────────────────
def main() -> None:
    print(f"{DIM}Opening a new case file...{RST}")
    case = write_case()
    system = suspect_system(case)

    print(f"\n{BOLD}{case['title'].upper()}{RST}\n{case['briefing']}\n")
    print(f"{BOLD}Suspect:{RST} {case['suspect_name']}")
    print(f"{BOLD}Your evidence:{RST}")
    for ev in case["detective_evidence"]:
        print(f"  • {ev}")
    print(
        f"\n{DIM}You have {MAX_QUESTIONS} questions. Lying doesn't mean guilty, so work out what the lies protect."
        f"\nCommands: /evidence  /accuse  /quit{RST}"
    )

    history: list = []
    transcript: list = []  # (question, answer) pairs for the polygraph
    ledger: list = []  # every lie with turn number and whether it was pushed on
    stress, notes_sent = 10, 0
    print(pulse_bar(stress))

    turn = 0
    while turn < MAX_QUESTIONS:
        try:
            q = input(f"\n{BOLD}You ({MAX_QUESTIONS - turn} left) ▸ {RST}").strip()
        except (EOFError, KeyboardInterrupt):
            q = "/quit"
        if not q:
            continue
        if q == "/quit":
            reveal(case, ledger, None, "")
            return
        if q == "/evidence":
            for ev in case["detective_evidence"]:
                print(f"  • {ev}")
            continue
        if q == "/accuse":
            break

        turn += 1
        note = None
        if notes_sent < len(STRESS_NOTES) and stress >= STRESS_NOTES[notes_sent][0]:
            note = STRESS_NOTES[notes_sent][1]
            notes_sent += 1

        answer = ask_suspect(system, history, q, note)
        p = polygraph(case, transcript, q, answer)
        transcript.append((q, answer))

        # a lie counts as "pushed on" if a later question pressed hard on it
        if p["pressure"] >= 2 or p["caught_contradiction"]:
            for entry in ledger:
                if not entry["pushed"] and entry["turn"] >= turn - 2:
                    entry["pushed"] = True

        lies_by_id = {l["id"]: l for l in case["planned_lies"]}
        seen = {e["id"] for e in ledger}
        for lid in p["planned_lie_ids"]:
            if lid in lies_by_id and lid not in seen:
                ledger.append({"id": lid, "turn": turn, "lie": lies_by_id[lid]["claim"],
                               "truth": lies_by_id[lid]["truth"], "pushed": False})
        if p["improvised_lie"]:
            ledger.append({"id": f"X{turn}", "turn": turn, "lie": p["improvised_lie"],
                           "truth": "(improvised on the spot)", "pushed": False})

        stress += p["pressure"] * 7 + (15 if p["caught_contradiction"] else 0) + (4 if p["lied"] else 0) - 3
        stress = max(0, min(100, stress))
        print(pulse_bar(stress))

        if p["confessed"]:
            print(f"\n{GRN}{BOLD}They cracked. You got a confession.{RST}")
            reveal(case, ledger, case["guilty"], case["secret"])
            return

    print(f"\n{BOLD}Time to decide.{RST}")
    g = input("Guilty of the crime? (y/n) ▸ ").strip().lower().startswith("y")
    theory = input("What were they really hiding? (one line) ▸ ").strip()
    reveal(case, ledger, g, theory)


if __name__ == "__main__":
    run(main)
