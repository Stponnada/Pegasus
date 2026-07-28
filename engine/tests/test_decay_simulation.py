"""Layer 6 verification (spec §10): simulate strength evolution over a 30-day
period with a daily decay job, and confirm per-class behaviour + dormancy."""

from datetime import timedelta

from ontomem.embeddings import HashingEmbedder
from ontomem.engine import Engine, _now
from ontomem.extractor import ExtractionResult
from ontomem.merge import NEW
from ontomem.merge_llm import DisambiguationDecision
from ontomem.model import Edge, Episode, Node


def _engine_with_edge(tmp_path, stability):
    extraction = ExtractionResult(
        episode=Episode.create("seed", 0.5),
        nodes=[Node.create("PERSON", "Bill"), Node.create("ORG", "Walmart")],
        edges=[Edge.create("PERSON::bill", "WORKS_AT", "ORG::walmart", stability=stability)],
    )
    eng = Engine(
        tmp_path, embedder=HashingEmbedder(),
        extract_fn=lambda c, ctx: extraction,
        disambiguate_fn=lambda e, r, s, c: DisambiguationDecision(e.key, NEW, None, 0.0),
    )
    eng.write([{"role": "user", "text": "x"}])
    return eng


def _run_daily_decay(eng, days):
    key = "PERSON::bill::WORKS_AT::ORG::walmart"
    start = _now()
    eng.store.get_edge(key).updated_at = start.isoformat()
    for day in range(1, days + 1):
        eng.decay(now=start + timedelta(days=day))
    return eng.store.get_edge(key).strength


def test_stable_edge_barely_decays_over_30_days(tmp_path):
    eng = _engine_with_edge(tmp_path, "stable")  # lambda 0.005
    final = _run_daily_decay(eng, 30)
    # 100 * e^(-0.005*30) = 86.07
    assert 85 < final < 87


def test_mutable_edge_decays_noticeably(tmp_path):
    eng = _engine_with_edge(tmp_path, "mutable")  # lambda 0.020
    final = _run_daily_decay(eng, 30)
    # 100 * e^(-0.02*30) = 54.88
    assert 53 < final < 57


def test_ephemeral_edge_goes_dormant(tmp_path):
    eng = _engine_with_edge(tmp_path, "ephemeral")  # lambda 0.200
    final = _run_daily_decay(eng, 30)
    # 100 * e^(-0.2*30) ~ 0.25  -> well below dormancy threshold 2.0
    assert final < 2.0
    # but never deleted
    assert eng.store.get_edge("PERSON::bill::WORKS_AT::ORG::walmart") is not None


def test_daily_decay_matches_single_step(tmp_path):
    # multiplicative decay: 30 daily steps == one 30-day step
    eng = _engine_with_edge(tmp_path, "mutable")
    daily = _run_daily_decay(eng, 30)

    eng2 = _engine_with_edge(tmp_path / "b", "mutable")
    key = "PERSON::bill::WORKS_AT::ORG::walmart"
    start = _now()
    eng2.store.get_edge(key).updated_at = start.isoformat()
    eng2.decay(now=start + timedelta(days=30))
    single = eng2.store.get_edge(key).strength

    assert abs(daily - single) < 0.01


def test_reinforcement_keeps_edge_alive(tmp_path):
    # an edge reinforced periodically stays strong vs. one left to decay
    eng = _engine_with_edge(tmp_path, "mutable")
    key = "PERSON::bill::WORKS_AT::ORG::walmart"
    start = _now()
    eng.store.get_edge(key).updated_at = start.isoformat()
    for day in range(1, 31):
        eng.decay(now=start + timedelta(days=day))
        if day % 5 == 0:  # reinforced every 5 days
            eng.store.reinforce_edge(key, boost=15.0)
            # reinforcement at this simulated day resets the decay baseline to it
            eng.store.get_edge(key).updated_at = (start + timedelta(days=day)).isoformat()
    assert eng.store.get_edge(key).strength > 80  # stays healthy
