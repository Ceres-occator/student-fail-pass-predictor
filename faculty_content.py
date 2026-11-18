"""Plain-language wording shared by the faculty-facing apps (Streamlit and Dash).

Everything a teacher reads on screen lives here, so both apps describe a student
the same way and no technical terms leak into the interface.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from train_model import FEATURE_COLS

APP_NAME = "Student Success Check"
APP_TAGLINE = (
    "See at a glance which students are on track and which may need an early check-in, "
    "based on their online class activity this term."
)

# Activities grouped the way a teacher thinks about them.
PARTICIPATION_COLS = ["Days.Logged.In", "Read.Actions", "Forum.Actions", "Quiz.Actions"]
COURSEWORK_COLS = ["Submission.Actions", "Create.Actions", "Update.Actions", "Delete.Actions"]

ACTIVITY_INFO: dict[str, dict[str, str]] = {
    "Submission.Actions": {
        "label": "Assignments submitted",
        "help": "How many times the student turned in homework, projects, or other coursework.",
    },
    "Quiz.Actions": {
        "label": "Quiz activity",
        "help": "How many times the student opened, answered, or submitted online quizzes.",
    },
    "Create.Actions": {
        "label": "New work started",
        "help": "New drafts, entries, or uploads the student started in the class portal.",
    },
    "Update.Actions": {
        "label": "Edits to their work",
        "help": "How often the student went back to revise or improve their work.",
    },
    "Delete.Actions": {
        "label": "Items deleted",
        "help": "Drafts or files the student removed. Usually zero; it has little effect on the result.",
    },
    "Days.Logged.In": {
        "label": "Days active online",
        "help": "Number of different days the student signed in to the class portal.",
    },
    "Read.Actions": {
        "label": "Course materials opened",
        "help": "Lecture notes, modules, readings, and other pages the student viewed.",
    },
    "Forum.Actions": {
        "label": "Discussion activity",
        "help": "Posts, replies, and views on the class discussion board.",
    },
}

# Activities where "more than a typical passing student" is the healthy direction.
# Deletions are left out on purpose: fewer deletions is not a warning sign.
TRACKED_COLS = [c for c in FEATURE_COLS if c != "Delete.Actions"]

FOLLOW_UP_TIPS: dict[str, str] = {
    "Submission.Actions": "Check for missing homework or projects and agree on a plan for upcoming deadlines.",
    "Create.Actions": "Encourage starting work earlier and using office hours when stuck.",
    "Read.Actions": "Suggest setting aside regular time each week for the lecture notes and readings.",
    "Days.Logged.In": "Encourage logging in a few times a week rather than all at once before deadlines.",
    "Quiz.Actions": "Recommend practice quizzes to review key topics before graded exams.",
    "Forum.Actions": "Invite the student to ask questions or join the class discussion board.",
    "Update.Actions": "Encourage reviewing feedback and revising work before final submission.",
}


def _label(col: str) -> str:
    return ACTIVITY_INFO[col]["label"]


def certainty_text(chance: float) -> tuple[str, str]:
    """Describe how clear-cut the result is, without statistical terms."""
    distance = abs(chance - 0.5)
    if distance >= 0.3:
        return "Clear result", "The activity pattern strongly matches one group of past students."
    if distance >= 0.15:
        return "Fairly clear result", "The activity pattern leans clearly in one direction."
    return "Borderline result", "This student could go either way, so rely more on your own judgement."


def describe_result(result: dict[str, Any]) -> dict[str, Any]:
    """Turn a raw prediction into the wording shown to teachers."""
    on_track = result["prediction"] == 1
    chance = float(result["prob_pass"])
    if on_track:
        headline = "On track to pass"
        explanation = (
            "This student's online activity looks like that of past students who passed the course."
        )
    else:
        headline = "May need extra support"
        explanation = (
            "This student's online activity looks like that of past students who struggled. "
            "An early, friendly check-in is recommended."
        )
    return {
        "on_track": on_track,
        "status": "on-track" if on_track else "needs-support",
        "headline": headline,
        "explanation": explanation,
        "chance_pct": round(chance * 100),
        "certainty": certainty_text(chance)[0],
        "certainty_detail": certainty_text(chance)[1],
    }


def activity_snapshot(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare each activity with a typical passing student, in teacher terms."""
    rows: list[dict[str, Any]] = []
    for comp in result["comparisons"]:
        col = comp["feature"]
        typical = int(round(comp["high_median"]))
        value = int(comp["student_value"])
        if col not in TRACKED_COLS:
            status = "info"
            status_text = "For reference"
        elif value >= typical:
            status = "good"
            status_text = "On track"
        else:
            status = "low"
            status_text = "Below typical"
        rows.append(
            {
                "feature": col,
                "label": _label(col),
                "value": value,
                "typical": typical,
                "status": status,
                "status_text": status_text,
            }
        )
    order = {"low": 0, "good": 1, "info": 2}
    rows.sort(key=lambda r: (order[r["status"]], FEATURE_COLS.index(r["feature"])))
    return rows


def follow_up_suggestions(result: dict[str, Any]) -> list[dict[str, str]]:
    """Suggested next steps for the teacher. Each item has a short title and detail."""
    low = [r for r in activity_snapshot(result) if r["status"] == "low"]
    if result["prediction"] == 1:
        items = [
            {
                "title": "Keep it up",
                "text": "Acknowledge the student's steady effort and encourage the same habits.",
            }
        ]
        for r in low[:2]:
            items.append(
                {
                    "title": f"Keep an eye on {r['label'].lower()}",
                    "text": (
                        f"Currently {r['value']}; passing students usually have about {r['typical']}."
                    ),
                }
            )
        return items

    items = [
        {
            "title": r["label"],
            "text": (
                f"Currently {r['value']}; passing students usually have about {r['typical']}. "
                f"{FOLLOW_UP_TIPS.get(r['feature'], '')}"
            ).strip(),
        }
        for r in low
    ]
    if not items:
        items.append(
            {
                "title": "Look beyond the numbers",
                "text": (
                    "Activity levels are similar to passing students, so check the quality and "
                    "timeliness of recent work."
                ),
            }
        )
    items.append(
        {
            "title": "Reach out early",
            "text": (
                "A short one-on-one conversation, or a referral to tutoring or academic support, "
                "can make a real difference."
            ),
        }
    )
    return items


def influence_levels(feature_importances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank activities by how much they shape the result, as Strong / Moderate / Minor."""
    top = max(item["importance"] for item in feature_importances) or 1.0
    rows = []
    for item in sorted(feature_importances, key=lambda i: i["importance"], reverse=True):
        share = item["importance"] / top
        if share >= 0.6:
            level = "Strong"
        elif share >= 0.3:
            level = "Moderate"
        else:
            level = "Minor"
        rows.append(
            {
                "label": _label(item["feature"]),
                "help": ACTIVITY_INFO[item["feature"]]["help"],
                "level": level,
                "bar": round(share * 100),
            }
        )
    return rows


def reference_rows(feature_stats: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Typical activity levels for passing vs. struggling students."""
    return [
        {
            "Activity": _label(col),
            "What it means": ACTIVITY_INFO[col]["help"],
            "Typical passing student": int(round(feature_stats[col]["high_median"])),
            "Typical student who struggled": int(round(feature_stats[col]["low_median"])),
        }
        for col in FEATURE_COLS
    ]


def summary_record(
    name: str, course: str, inputs: dict[str, int], result: dict[str, Any]
) -> dict[str, Any]:
    """One row for the downloadable summary sheet."""
    desc = describe_result(result)
    return {
        "Student": name or "Unnamed student",
        "Course / Section": course or "Not specified",
        "Checked on": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Result": desc["headline"],
        "Chance of passing": f"{desc['chance_pct']}%",
        "How clear": desc["certainty"],
        **{_label(c): int(inputs[c]) for c in FEATURE_COLS},
        "Suggested follow-up": " | ".join(
            f"{s['title']}: {s['text']}" for s in follow_up_suggestions(result)
        ),
    }
