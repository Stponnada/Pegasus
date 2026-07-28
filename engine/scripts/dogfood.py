"""Dogfood harness: hold real multi-turn conversations with Gemini as a single
curious user, consolidate each into the memory graph (persisted across the
batch), then probe retrieval quality. Writes a full report to dogfood_report.json
for offline inspection.

Run from engine/:  PYTHONPATH=src uv run python scripts/dogfood.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google import genai

from ontomem.embeddings import GeminiEmbedder, HashingEmbedder
from ontomem.engine import Engine
from ontomem.genai_keys import call_rotating

MODEL = "gemini-3.1-flash-lite"


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# Offline embeddings (lexical seeding) sidestep the free-tier embedding quota so
# we can inspect EXTRACTION/MERGE quality (real Gemini) without an embed wall.
OFFLINE_EMBED = os.environ.get("ONTOMEM_OFFLINE_EMBED") == "1"
TRANSCRIPT_CACHE = Path(__file__).resolve().parents[1] / "dogfood_transcripts.json"


def with_backoff(fn, *, tries=6, base=4.0):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # rate limits / transient
            if attempt == tries - 1:
                raise
            wait = base * (2**attempt)
            print(f"   (retry {attempt+1}/{tries} after {wait:.0f}s: {str(exc)[:80]})")
            time.sleep(wait)


def assistant_reply(history: list[dict]) -> str:
    transcript = "\n".join(f"{t['role'].capitalize()}: {t['text']}" for t in history)
    prompt = (
        "You are a warm, knowledgeable conversational assistant. Continue this "
        "conversation with a natural reply of 2-4 sentences. Be substantive and "
        "ask a follow-up question when it fits.\n\n" + transcript + "\nAssistant:"
    )
    def attempt(key: str):
        return genai.Client(api_key=key).models.generate_content(model=MODEL, contents=prompt)

    resp = with_backoff(lambda: call_rotating(attempt))
    return (resp.text or "").strip()


# Single coherent user across all conversations: an Anthropic engineer who loves
# Roman history, with friends, a partner, and a live career dilemma.
CONVERSATIONS = [
    ("work_intro", [
        "Hey! I just wanted to talk through some stuff. I'm Marcus, I'm a software engineer at Anthropic — I work on inference infrastructure, mostly making the models serve faster.",
        "It's good but intense. My manager Dana is great but the on-call rotation is brutal — I had three pages last night. My teammate Wei usually covers for me but he's on vacation.",
        "Yeah exactly. I keep telling myself I'll set boundaries but I never do. I think I just really care about the systems staying up.",
    ]),
    ("roman_history", [
        "On a totally different note — the way I unwind is reading about the Roman empire. Kind of obsessively, honestly. My friend Priya got me into it a couple years ago.",
        "Right now I'm fascinated by the Crisis of the Third Century — how Rome almost collapsed and then Diocletian basically rebuilt the whole administrative system. I find the tetrarchy genuinely brilliant.",
        "Do you think Diocletian's reforms actually saved the empire, or just delayed the inevitable? I go back and forth on this.",
        "That makes sense. Priya argues it just delayed it. I think I'm coming around to her view actually.",
    ]),
    ("personal_dilemma", [
        "Something's been on my mind. My partner Elena and I are planning a trip to Rome next spring — first big trip together. I'm really excited but also nervous about taking the time off.",
        "And there's a work thing tangled up in it. I got offered a transfer to the alignment safety team at Anthropic. It's the work I find most meaningful, but it'd mean leaving Dana's team and probably more uncertainty.",
        "I think deep down I want the safety role. The infrastructure work is comfortable but the safety mission is why I joined Anthropic in the first place. I'm just scared of the change.",
    ]),
    ("followup_callback", [
        "Hey, back again. I've been thinking more about that big decision at work I mentioned.",
        "Yeah, the team transfer. I think I'm going to go for it. Also Elena and I booked the flights for the trip, so that's locked in now!",
    ]),
]


def run_conversation(title: str, human_turns: list[str]) -> list[dict]:
    print(f"\n=== conversation: {title} ===")
    history: list[dict] = []
    turn = 0
    for human in human_turns:
        turn += 1
        history.append({"role": "user", "turn": turn, "text": human})
        print(f"  USER: {human[:90]}")
        reply = assistant_reply(history)
        turn += 1
        history.append({"role": "assistant", "turn": turn, "text": reply})
        print(f"  ASST: {reply[:90]}")
        time.sleep(1.0)
    return history


def dump_graph(engine: Engine) -> dict:
    return {
        "nodes": [
            {"key": n.key, "name": n.name, "kind": n.kind, "aliases": n.aliases, "properties": n.properties}
            for n in engine.store.nodes.values()
        ],
        "edges": [
            {"edge": f"{e.source_key} -[{e.relation}]-> {e.target_key}", "stability": e.stability,
             "strength": round(e.strength, 1), "confidence": e.confidence, "snippet": e.snippet}
            for e in engine.store.edges.values()
        ],
        "episodes": [
            {"summary": ep.summary, "importance": ep.importance, "tags": ep.tags}
            for ep in engine.store.episodes.values()
        ],
    }


PROBES = [
    "I'm getting ready for that trip to Rome — what should I think about packing?",
    "Remind me what the career decision was that I've been wrestling with.",
    "What do Priya and I usually talk about?",
    "How's the on-call situation been treating me?",
    "Tell me something about Diocletian.",
    "What was my partner's name again?",
]


def load_or_run_conversations() -> list[dict]:
    """Generate the conversations once and cache them, so reruns (for engine
    tuning) don't waste generation calls re-creating assistant turns."""
    if TRANSCRIPT_CACHE.exists():
        print(f"(using cached transcripts: {TRANSCRIPT_CACHE.name})")
        return json.loads(TRANSCRIPT_CACHE.read_text())
    convos = [{"title": title, "transcript": run_conversation(title, turns)} for title, turns in CONVERSATIONS]
    TRANSCRIPT_CACHE.write_text(json.dumps(convos, indent=2, ensure_ascii=False))
    return convos


def main():
    out_dir = Path(__file__).resolve().parents[1] / "dogfood_data"
    report_path = Path(__file__).resolve().parents[1] / "dogfood_report.json"
    embedder = HashingEmbedder() if OFFLINE_EMBED else GeminiEmbedder()
    print(f"embedder: {embedder.model_id}")
    engine = Engine(out_dir, embedder=embedder)

    report = {"conversations": [], "writes": [], "probes": [], "retrieve_memory": []}

    for convo in load_or_run_conversations():
        title, transcript = convo["title"], convo["transcript"]
        report["conversations"].append({"title": title, "transcript": transcript})
        print(f"  -> writing '{title}' to graph...")
        result = with_backoff(lambda: engine.write(transcript))
        report["writes"].append({"title": title, "result": result})
        print(f"     {json.dumps({k: result[k] for k in ('nodes_created','nodes_merged','edges_created','edges_reinforced','edges_dropped','flagged','warnings')})}")
        time.sleep(1.0)

    report["graph"] = dump_graph(engine)
    print(f"\n=== final graph: {len(engine.store.nodes)} nodes, {len(engine.store.edges)} edges ===")

    print("\n=== retrieval probes ===")
    for q in PROBES:
        r = with_backoff(lambda: engine.read(q))
        report["probes"].append({"query": q, "memory_block": r["memory_block"],
                                 "context_block": r["context_block"], "trace": r["trace"]})
        print(f"\nQ: {q}")
        print(r["text"] if r["text"] else "  (nothing fired)")
        time.sleep(1.0)

    for name in ["Marcus", "Elena", "Diocletian", "Anthropic"]:
        rm = engine.retrieve_memory(name, depth=2)
        report["retrieve_memory"].append({"name": name, "result": rm})

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nfull report -> {report_path}")


if __name__ == "__main__":
    main()
