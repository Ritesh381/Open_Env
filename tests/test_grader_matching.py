"""Regression tests for grader matching and weighted scoring."""

import json
from pathlib import Path

import pytest

from pr_review_env import Action, InlineComment, ReviewDecision
from pr_review_env.models import GroundTruth
from pr_review_env.server.grader import ReviewGrader


def _load_task_by_filename(filename: str) -> dict:
    root = Path(__file__).resolve().parent.parent
    with open(root / "tasks" / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def test_task3_maintainability_issues_use_distinct_lines():
    task = _load_task_by_filename("task3_advanced_review.json")
    issues = task["pr_scenario"]["ground_truth"]["issues"]
    tp_lines = [
        i["line"]
        for i in issues
        if i["file"] == "workers/task_processor.py" and i["category"] == "maintainability"
    ]
    assert sorted(tp_lines) == [83, 84]


def test_weighted_score_includes_severity_from_task_config():
    grader = ReviewGrader()
    weights = {
        "precision": 0.4,
        "recall": 0.4,
        "coverage": 0.0,
        "severity": 0.2,
    }
    s_high = grader._calculate_weighted_score(1.0, 1.0, 1.0, 1.0, weights)
    s_low = grader._calculate_weighted_score(1.0, 1.0, 1.0, 0.0, weights)
    assert s_high == pytest.approx(1.0)
    assert s_low == pytest.approx(0.8)
    assert s_high > s_low


def test_duplicate_comments_same_line_one_issue_becomes_fp():
    """Two comments targeting the same ground-truth row: only one can match."""
    task = _load_task_by_filename("task7.json")
    grader = ReviewGrader()
    gt = GroundTruth(**task["pr_scenario"]["ground_truth"])
    path = "content/features/liveStreamRecorder/liveStreamRecorder.js"
    dup = Action(
        inline_comments=[
            InlineComment(
                file_path=path,
                line_number=83,
                comment="MutationObserver with subtree true causes expensive layout thrash in this SPA.",
                severity="warning",
                category="performance",
            ),
            InlineComment(
                file_path=path,
                line_number=83,
                comment="Observing subtree on the app root is costly; narrow the observer target.",
                severity="warning",
                category="performance",
            ),
        ],
        general_comments=[],
        decision=ReviewDecision(decision="request_changes", summary="Performance."),
        submit=True,
    )
    fb = grader.grade_review(dup, gt, task)
    assert fb.true_positives == 1
    assert fb.false_positives == 1


def test_build_prompt_includes_hunk_section():
    from inference import build_prompt

    task = _load_task_by_filename("task1_security_basic.json")
    pr = task["pr_scenario"]
    text = build_prompt(pr, "task1_security_basic")
    assert "Hunks (full unified diff):" in text
    assert "@@ lines" in text
