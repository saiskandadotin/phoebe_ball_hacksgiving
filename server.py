"""Phoebe Ball — a weird little game server.

Questions come from Xiaomimimo (Anthropic-compatible mimo endpoint);
answers are rated 0-10 by Jev via the TypeSafeAI sprite connector.
"""

import json
import re
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

XIAOMIMIMO_MESSAGES_URL = (
    "https://api.sprites.dev/v1/gateway/custom_api/GORWA85ekzxuBuaLkCo-Xg/v1/messages"
)
TYPESAFE_SYSTEMONE_URL = (
    "https://api.sprites.dev/v1/gateway/typesafe/vOGFjSiOMAE09y2NdL2Lbw/v1/systemone"
)
MIMO_MODEL = "mimo-v2.6-flash"
JEV_MODEL = "jev-latest"

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Phoebe Ball", docs_url=None, redoc_url=None)

QUESTION_SYSTEM_PROMPT = """\
Ask ONE delightfully weird, whimsical, question for the player
Ask a single question - one sentence maximum

Examples:
- what is your favourite thing about trees?
- What would your smelly cat say about you if it could text?
- if you caould rename yourself, what would it be?
- What is your favorite thing about elbows?
- Which kitchen utensil has the saddest aura?
- What do you think pigeons are actually whispering about?
- On a scale of Tuesday to purple, how honest are you feeling right now?
- What is the most trustworthy vegetable?
- What does the color yellow smell like to you?
- What is your favorite sound that a lamp makes?
- If you were a ghost, what mildly inconvenient thing would you haunt?
- What is the best thing about rocks?
- Which letter of the alphabet looks like it owes people money?
- What do you think dust bunnies dream about?
- If spoons and forks went to war, who would the butter knives side with?

Respond with ONLY a JSON object, no markdown fences, no extra text: {"question": "<the question>"}\
"""

# Jev's `score` type is an ordinal scale: the returned score is the
# probability-weighted index into this anchor list (max 10 anchors), so we
# use 10 ordered anchors and rescale index 0-9 -> 0-10 for the user.
JEV_ANCHORS = [
    "0 - phoebe does not like it at all",
    "1 - phoebe does not like it",
    "2 - phoebe does not really like it",
    "3 - phoebe is not a fan",
    "4 - neutral, a shrug",
    "5 - neutral, mildly amused",
    "6 - neutral, slightly charmed",
    "7 - phoebe likes it",
    "8 - phoebe really likes it",
    "9 - phoebe absolutely loves it",
]


class QuestionRequest(BaseModel):
    asked: list[str] = []


class RateRequest(BaseModel):
    question: str
    answer: str


def _text_from_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""


def _extract_json(text: str) -> dict:
    text = text.strip()
    # Strip accidental markdown fences.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"No JSON object found in: {text[:200]}")


def _score_label(score: int) -> str:
    if score <= 3:
        return "phoebe does not like"
    if score <= 6:
        return "neutral"
    return "loves the answer"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/question")
async def make_question(req: QuestionRequest):
    asked_block = ""
    if req.asked:
        listed = "\n".join(f"- {q}" for q in req.asked[-10:])
        asked_block = f"\n\nAlready asked (do not repeat):\n{listed}"

    payload = {
        "model": MIMO_MODEL,
        "max_tokens": 2048,
        "system": QUESTION_SYSTEM_PROMPT + asked_block,
        "messages": [{"role": "user", "content": "Ask me the question now."}],
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                XIAOMIMIMO_MESSAGES_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
            )
        resp.raise_for_status()
        data = resp.json()
        text = _text_from_content(data.get("content", ""))
        question = _extract_json(text).get("question", "").strip()
        if not question:
            raise ValueError("Empty question")
        return {"question": question}
    except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError) as exc:
        return JSONResponse(
            status_code=502,
            content={"error": f"Phoebe lost her train of thought: {exc}"},
        )


@app.post("/api/rate")
async def rate_answer(req: RateRequest):
    state = (
        "Phoebe Ball game, hosted by Phoebe Buffay from the TV show Friends. "
        "Judge the player's answer on how much Phoebe would like it.\n"
        f"Question asked: {req.question}\n"
        f"Player's answer: {req.answer}"
    )
    payload = {
        "model": JEV_MODEL,
        "state": state,
        "questions": {
            "rating": {
                "type": "score",
                "criteria": JEV_ANCHORS,
            }
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                TYPESAFE_SYSTEMONE_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        resp.raise_for_status()
        data = resp.json()
        raw = float(data["answers"]["rating"]["score"])
        # raw is a weighted index over the 10 anchors (0-9); rescale to 0-10.
        scaled = raw * 10.0 / 9.0
        score = max(0, min(10, int(round(scaled))))
        return {
            "score": score,
            "raw": raw,
            "label": _score_label(score),
            "confidence": data["answers"]["rating"].get("confidence"),
        }
    except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        return JSONResponse(
            status_code=502,
            content={"error": f"Jev is pondering too hard: {exc}"},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
