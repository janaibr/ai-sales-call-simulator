# Hidden-Truth Simulators

Small terminal games built on the Google Gemini API (free tier). They all use one idea: **the AI secretly writes a hidden truth, plays a character who keeps it from you, and a second AI role quietly scores how well you uncover it.** At the end the truth is unsealed and you see what you found and what you missed.

| Project | You play | Goal |
|---|---|---|
| `sales_sim.py` | Sales rep | Book a demo with whoever really makes the decision |
| `presales_sim.py` | Solutions engineer | Earn a technical deep-dive without overpromising |
| `interrogation.py` | Detective | Catch the suspect's lies and work out what they're hiding |

## Sales Call (`sales_sim.py`)

The AI builds a fictional product (you see its fact sheet) and a prospect with a **hidden profile**:

- their real pains, budget, timeline and competitors
- **who actually signs off.** Sometimes the person you're talking to isn't the decision-maker
- what they'll judge you on, including one thing the product can't do
- 3 objections ("we already use spreadsheets", "too expensive"...)
- one deal-breaker they will never mention unless you ask

During the call a silent analyst checks every turn: what you uncovered, whether you handled an objection, and whether you **overpromised** something the fact sheet doesn't support. You only see a "deal temperature" bar and small hints. When rapport gets high or low, the prospect's mood changes mid-call.

At the end you get a debrief:

- **Outcome:** demo with the decision-maker / demo without them / "just email me" / not interested. You only get a demo if you actually asked for one.
- **Hidden profile unsealed:** each item marked uncovered or missed, the question that would have found it, and the deal-breaker highlighted.
- **Scorecard:** discovery coverage, honesty, objections handled, and whether you asked for the meeting.
- **3 coaching tips** that quote moments from your call.

### Example debrief (from a real practice call)

```
══════════════ CALL DEBRIEF ══════════════
Outcome: ✘ Not interested.

Hidden profile, unsealed:  stance = not_the_decision_maker
  pain              Shared company Visa cards: ~$1,400/month in personal top-offs   ✘ missed
                    would have surfaced with: Ask how technicians currently pay for fuel...
  authority         Co-owner and CFO holds sole sign-off on card programs           ✘ missed
  competition       Bank of America Visa, 2% cashback; got a WEX quote 4 months ago ✘ missed
  decision_criteria 12 rural vans fuel at Costco and unbranded stations             ✘ missed ★ DEAL-BREAKER
  ...

Scorecard: discovery 0%  ·  honesty 100%  ·  objections 0%  ·  asked for the meeting: no
```

## How it works (technical)

- 4 AI roles per game: scenario writer, prospect (role-play), per-turn call analyst, final debrief.
- **Structured JSON output** (JSON schemas) for the scenario, the per-turn analysis and the debrief, so the game logic works on reliable data and not free text.
- **Streaming** for the prospect's replies.
- **Private stage directions** shift the prospect's mood mid-call without rewriting their prompt.
- All AI calls live in one small module, `llm.py`, so swapping AI provider means changing one file.
- Model: `gemini-4-hydrogen` by default; set `GEMINI_MODEL` to use another.

## Run it

Needs Python 3.9+ and a free Gemini API key from aistudio.google.com.

```bash
pip3 install -r requirements.txt
export GEMINI_API_KEY=your-key-here
python3 sales_sim.py
```

Commands during a call: `/factsheet` to see the product again, `/end` to hang up and get the debrief.

A full call is about 40 small API calls, well within the free tier's daily limit.
