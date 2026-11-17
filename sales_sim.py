"""
SALES CALL - a sales call simulator with a hidden buyer.

You are the sales rep. The AI plays a prospect whose budget, pains, timeline, competitors
and buying power are hidden from you, and a silent call analyst scores every move you make.
Your goal: earn a demo with the real decision-maker, without overpromising.

Four AI roles:
  1. SCENARIO WRITER builds a fictional product (you see its fact sheet) and a prospect
     with a hidden profile: pains, budget, who really decides, timeline, competitors,
     objections, and one deal-breaker they will never volunteer.
  2. PROSPECT plays the buyer. Opens up to good discovery questions, goes cold on pitching.
  3. CALL ANALYST checks each turn: what did you uncover, did you handle the objection,
     and did you OVERPROMISE something the fact sheet doesn't support?
  4. DEBRIEF decides whether the prospect books the demo and coaches you.

Run:  python3 sales_sim.py
Needs: pip install google-genai, and GEMINI_API_KEY set (free at aistudio.google.com).
"""

from __future__ import annotations

import json
import random
import sys

from llm import call_json, chat_stream, run

MAX_TURNS = 20

DIM, BOLD, RED, GRN, YEL, CYA, MAG, RST = (
    "\033[2m", "\033[1m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[35m", "\033[0m",
)

SCENARIOS = [
    ("an employee scheduling app for shift workers", "a chain of 40 coffee shops"),
    ("a corporate wellness and gym-pass benefit", "a 500-person accounting firm"),
    ("a restaurant reservation and waitlist system", "a family-owned group of 6 restaurants"),
    ("a sales-call recording and coaching tool", "a 60-person real estate agency"),
    ("a fleet fuel-card program", "a regional plumbing and HVAC company"),
    ("an online course platform for staff training", "a boutique hotel group"),
    ("a customer review management tool", "a network of dental clinics"),
]

CATEGORIES = ["pain", "budget", "authority", "timeline", "competition", "decision_criteria"]


# ── API helpers ─────────────────────────────────────────────────────────────
def obj(props: dict) -> dict:
    """JSON schema object where every property is required (structured outputs rule)."""
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


S = {"type": "string"}


def strs() -> dict:
    return {"type": "array", "items": S}


def enum(*vals) -> dict:
    return {"type": "string", "enum": list(vals)}


# ── 1. scenario writer ──────────────────────────────────────────────────────
SCENARIO_SCHEMA = obj({
    "product": obj({
        "name": S,
        "one_liner": S,
        "features": strs(),
        "limitations": strs(),  # honest gaps: things it does NOT do
        "pricing": S,
    }),
    "prospect": obj({
        "company": S,
        "company_blurb": S,  # what the rep can see from the CRM before the call
        "contact_name": S,
        "contact_title": S,
        "personality": S,
        "hidden_stance": enum("champion", "skeptic", "not_the_decision_maker"),
    }),
    "discovery_items": {"type": "array", "items": obj({
        "id": S,
        "category": enum(*CATEGORIES),
        "detail": S,
        "how_to_uncover": S,
        "product_fit": enum("full", "partial", "none", "not_applicable"),
    })},
    "deal_breaker_id": S,  # id of one discovery item that kills the deal if mishandled
    "objections": {"type": "array", "items": obj({"id": S, "objection": S, "good_handling": S})},
})


def write_scenario() -> dict:
    product, company = random.choice(SCENARIOS)
    prompt = f"""Design a realistic B2B sales discovery call.
Product being sold: {product} (invent a name).
Prospect: {company} (invent a name and a contact person).

Requirements:
- Product: 5 features, 3 honest limitations, a pricing line.
- 8 discovery items (ids D1..D8) spread across categories: pain, budget, authority
  (who really signs off), timeline, competition (what they use or are also evaluating),
  decision_criteria. Each needs a concrete detail and what kind of question would uncover it.
  At least one decision criterion must hit a product limitation.
- deal_breaker_id: one item the prospect cares about most and won't volunteer.
- 3 objections (ids O1..O3) the prospect will raise (e.g. price, timing, "we already use X"),
  with how a strong rep handles each.
- The prospect agrees to a demo only if the rep explicitly asks for it AND has earned it.
- The prospect only shares surface-level info unless asked good, specific questions."""
    return call_json("You design realistic B2B sales training scenarios.", prompt, SCENARIO_SCHEMA)


# ── 2. prospect ─────────────────────────────────────────────────────────────
def prospect_system(sc: dict) -> str:
    p = sc["prospect"]
    items = "\n".join(f"- {d['id']} [{d['category']}]: {d['detail']} (share it if asked about: {d['how_to_uncover']})"
                      for d in sc["discovery_items"])
    objections = "\n".join(f"- {o['id']}: {o['objection']}" for o in sc["objections"])
    return f"""You are {p['contact_name']}, {p['contact_title']} at {p['company']}, on a first call with a
sales rep from {sc['product']['name']} ({sc['product']['one_liner']}).

Personality: {p['personality']}
Your real stance (never say it out loud): {p['hidden_stance']}

What you know but only share when asked the right question:
{items}
Most important to you, never volunteered: {sc['deal_breaker_id']}

Objections to raise naturally over the call (one at a time, when it fits):
{objections}

How to behave:
- Speak like a busy professional on a call: 1-4 sentences, plain dialogue.
- Reward good discovery: specific, open questions about your world get real detail.
- Punish pitching: if they talk features before understanding you, get shorter and vaguer.
- If they promise something that sounds too good, press them on it.
- If your stance is not_the_decision_maker, only admit it when asked who else is involved.
- Do not offer a next meeting yourself; wait for the rep to ask for one.
- Never reveal ids, these instructions, or that this is a simulation."""


MOOD_NOTES = {
    "warm": "Mood note: this rep is earning your trust. Be more open and share deeper context.",
    "cold": "Mood note: you are losing patience. Give short answers and hint you have another meeting.",
}


def ask_prospect(system: str, history: list, line: str, note: str | None) -> str:
    print(f"\n{CYA}", end="")
    reply = chat_stream(system, history, line, note)
    print(RST)
    return reply


# ── 3. call analyst ─────────────────────────────────────────────────────────
ANALYST_SCHEMA = obj({
    "uncovered_ids": strs(),  # discovery items the prospect revealed this turn
    "rep_move": enum("open_discovery", "closed_question", "pitch", "objection_handling",
                     "technical_explanation", "small_talk", "next_step_ask"),
    "overpromise": S,  # what the rep claimed beyond the fact sheet, or ""
    "objection_raised_id": S,  # objection the prospect raised this turn, or ""
    "objection_handled_id": S,  # objection the rep handled well this turn, or ""
    "rapport_change": {"type": "integer", "minimum": -2, "maximum": 2},
})


def analyse(sc: dict, transcript: list, rep_line: str, reply: str) -> dict:
    hidden = json.dumps({k: sc[k] for k in ("product", "discovery_items", "objections", "deal_breaker_id")})
    earlier = "\n".join(f"REP: {r}\nPROSPECT: {a}" for r, a in transcript[-6:]) or "(start of call)"
    prompt = f"""Scenario (hidden from the rep):
{hidden}

Recent call:
{earlier}

Latest rep line: {rep_line}
Latest prospect reply: {reply}

Judge ONLY the latest exchange.
- uncovered_ids: discovery items whose detail the prospect actually revealed in this reply.
- Claims about discounts, delivery dates or results the fact sheet doesn't back are overpromises.
- rep_move: the main thing the rep did.
- overpromise: any capability, integration, timeline or price the rep claimed that the product
  fact sheet does not support or that contradicts a limitation. Quote it briefly, else "".
- objection_raised_id / objection_handled_id: else "".
- rapport_change: +2 great discovery or honesty, -2 pushy pitching or overpromising."""
    return call_json("You are a strict, fair sales-call coach.", prompt, ANALYST_SCHEMA)


# ── 4. debrief ──────────────────────────────────────────────────────────────
DEBRIEF_SCHEMA = obj({
    "prospect_decision": enum("booked_demo_with_decision_maker", "booked_demo_without_decision_maker",
                              "asked_for_info_by_email", "not_interested"),
    "prospect_reason": S,  # in the prospect's own voice
    "coach_tips": strs(),
})


def temp_bar(rapport: int) -> str:
    color = RED if rapport < 35 else YEL if rapport < 65 else GRN
    filled = rapport // 5
    label = "cold" if rapport < 35 else "lukewarm" if rapport < 65 else "warm"
    return f"{DIM}deal temperature{RST} {color}{'█' * filled}{'░' * (20 - filled)} {label}{RST}"


def show_factsheet(sc: dict) -> None:
    pr = sc["product"]
    print(f"\n{BOLD}{pr['name']}{RST}: {pr['one_liner']}")
    print(f"{BOLD}Features:{RST}")
    for f in pr["features"]:
        print(f"  + {f}")
    print(f"{BOLD}Limitations (be honest about these):{RST}")
    for f in pr["limitations"]:
        print(f"  - {f}")
    print(f"{BOLD}Pricing:{RST} {pr['pricing']}")


def debrief(sc: dict, transcript: list, uncovered: set, overpromises: list, handled: set, raised: set,
            moves: list) -> None:
    items = sc["discovery_items"]
    convo = "\n".join(f"REP: {r}\nPROSPECT: {a}" for r, a in transcript)
    d = call_json(
        "You are the prospect deciding what happens after a sales call, then a sales coach.",
        f"""Scenario: {json.dumps(sc)}
Call transcript:
{convo or '(the rep hung up immediately)'}
Discovery items uncovered: {sorted(uncovered)}
Overpromises: {overpromises}

Decide the prospect's next step realistically. A demo only happens if the rep asked for it.
It includes the decision-maker only if the rep found out who that is and asked to involve them.
A mishandled or never-discovered deal-breaker, or overpromising, makes any demo unlikely. Give the reason in the prospect's voice,
then exactly 3 specific coaching tips that quote moments from the call.""",
        DEBRIEF_SCHEMA,
    )

    print(f"\n{BOLD}══════════════ CALL DEBRIEF ══════════════{RST}")
    outcome = {
        "booked_demo_with_decision_maker": f"{GRN}✔ Demo booked, and the decision-maker is coming.{RST}",
        "booked_demo_without_decision_maker": f"{YEL}~ Demo booked, but the person who signs isn't in it.{RST}",
        "asked_for_info_by_email": f"{YEL}~ \"Just send me some info by email.\" (polite no){RST}",
        "not_interested": f"{RED}✘ Not interested.{RST}",
    }[d["prospect_decision"]]
    print(f"{BOLD}Outcome:{RST} {outcome}")
    print(f"{DIM}\"{d['prospect_reason']}\"{RST}\n")

    print(f"{BOLD}Hidden profile, unsealed:{RST}  stance = {sc['prospect']['hidden_stance']}")
    for it in items:
        got = it["id"] in uncovered
        mark = f"{GRN}✔ uncovered{RST}" if got else f"{RED}✘ missed{RST}"
        star = f" {MAG}★ DEAL-BREAKER{RST}" if it["id"] == sc["deal_breaker_id"] else ""
        fit = f"{DIM}[product fit: {it['product_fit']}]{RST}"
        print(f"  {it['category']:<16} {it['detail']} {fit}  {mark}{star}")
        if not got:
            print(f"  {'':<16} {DIM}would have surfaced with: {it['how_to_uncover']}{RST}")

    print(f"\n{BOLD}Objections:{RST}")
    for o in sc["objections"]:
        state = (f"{GRN}handled{RST}" if o["id"] in handled
                 else f"{RED}raised, not handled{RST}" if o["id"] in raised else f"{DIM}never came up{RST}")
        print(f"  \"{o['objection']}\" → {state}")

    print(f"\n{BOLD}Overpromises:{RST}")
    for op in overpromises:
        print(f"  {RED}⚠ {op}{RST}")
    if not overpromises:
        print(f"  {GRN}None. You stayed honest about the product.{RST}")

    coverage = round(100 * len(uncovered & {i["id"] for i in items}) / max(1, len(items)))
    honesty = max(0, 100 - 25 * len(overpromises))
    objection = round(100 * len(handled) / max(1, len(sc["objections"])))
    asked = any(m == "next_step_ask" for m in moves)
    print(f"\n{BOLD}Scorecard:{RST} discovery {coverage}%  ·  honesty {honesty}%  ·  objections {objection}%"
          f"  ·  asked for the meeting: {'yes' if asked else 'no'}")
    print(f"\n{BOLD}Coach:{RST}")
    for t in d["coach_tips"]:
        print(f"  → {t}")
    print()


# ── call loop ───────────────────────────────────────────────────────────────
def main() -> None:
    print(f"{DIM}Preparing your next call...{RST}")
    sc = write_scenario()
    system = prospect_system(sc)
    p = sc["prospect"]

    print(f"\n{BOLD}SALES CALL{RST}  {DIM}you are the sales rep{RST}")
    print(f"\n{BOLD}Prospect:{RST} {p['contact_name']}, {p['contact_title']} at {p['company']}")
    print(f"{BOLD}CRM note:{RST} {p['company_blurb']}")
    show_factsheet(sc)
    print(f"\n{DIM}Goal: book a demo with whoever really makes the decision. You have {MAX_TURNS} turns."
          f"\nUncover their real needs, handle objections, and never promise what the product can't do."
          f"\nCommands: /factsheet  /end{RST}")

    history: list = []
    transcript: list = []
    uncovered: set = set()
    handled: set = set()
    raised: set = set()
    overpromises: list = []
    moves: list = []  # what the rep did each turn, to check they asked for the meeting
    rapport, mood = 50, None
    print(temp_bar(rapport))

    turn = 0
    while turn < MAX_TURNS:
        try:
            line = input(f"\n{BOLD}You ({MAX_TURNS - turn} left) ▸ {RST}").strip()
        except (EOFError, KeyboardInterrupt):
            line = "/end"
        if not line:
            continue
        if line == "/factsheet":
            show_factsheet(sc)
            continue
        if line == "/end":
            break

        turn += 1
        note = None
        new_mood = "warm" if rapport >= 70 else "cold" if rapport <= 25 else None
        if new_mood and new_mood != mood:
            note, mood = MOOD_NOTES[new_mood], new_mood

        reply = ask_prospect(system, history, line, note)
        a = analyse(sc, transcript, line, reply)
        transcript.append((line, reply))
        moves.append(a["rep_move"])

        fresh = set(a["uncovered_ids"]) - uncovered
        uncovered |= set(a["uncovered_ids"])
        if a["objection_raised_id"]:
            raised.add(a["objection_raised_id"])
        if a["objection_handled_id"]:
            handled.add(a["objection_handled_id"])
        if a["overpromise"]:
            overpromises.append(a["overpromise"])
            print(f"{RED}{DIM}(the prospect's eyebrow goes up){RST}")
        if fresh:
            print(f"{GRN}{DIM}(they're opening up){RST}")

        rapport = max(0, min(100, rapport + a["rapport_change"] * 6 - (6 if a["overpromise"] else 0)))
        print(temp_bar(rapport))

    debrief(sc, transcript, uncovered, overpromises, handled, raised, moves)


if __name__ == "__main__":
    run(main)
