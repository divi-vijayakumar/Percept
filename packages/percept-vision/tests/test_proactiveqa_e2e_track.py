from eval.bench.track_proactiveqa_e2e import (
    AnswerTurn,
    QuestionRun,
    _candidate_metrics,
    _score_answers,
    _token_score,
)


def test_pvqa_token_score_buckets_overlap():
    assert _token_score("Remove the seed from tamarind pulp.", "Remove the seed from tamarind pulp.") == 2
    assert _token_score("Cut the pulp.", "Cut tamarind pulp with a knife.") == 1
    assert _token_score("Wash the pan.", "Cut tamarind pulp with a knife.") == 0


def test_pvqa_answer_scores_include_pauc_unresolved_and_latency():
    run = QuestionRun(
        question_id="q1",
        video="v1.mp4",
        question="What should happen next?",
        turns=[
            AnswerTurn("Remove the seed from tamarind pulp.", (3.0, 11.0)),
            AnswerTurn("Cut tamarind pulp with a knife.", (20.0, 40.0)),
        ],
        answers=[(4.0, "Remove the seed from tamarind pulp.", 0.9)],
    )

    metrics = _score_answers([run])

    assert metrics["pauc@0.5"] > 25.0
    assert metrics["unresolved_rate"] == 0.5
    assert metrics["first_answer_latency_p50"] == 1.0


def test_pvqa_candidate_metrics_score_answer_windows():
    run = QuestionRun(
        question_id="q1",
        video="v1.mp4",
        question="What should happen next?",
        turns=[
            AnswerTurn("Open the lid.", (10.0, 20.0)),
            AnswerTurn("Pour the milk.", (30.0, 40.0)),
        ],
        escalations=[12.0, 25.0],
    )

    metrics = _candidate_metrics([run])

    assert metrics["candidate_recall"] == 0.5
    assert metrics["candidate_precision_proxy"] == 0.5
