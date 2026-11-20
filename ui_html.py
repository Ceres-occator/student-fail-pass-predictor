"""HTML building blocks shared by the Streamlit and Dash apps.

Every function returns a single-line HTML string styled by assets/style.css, so both
apps render identical markup. Anything typed by a user is escaped here.
"""

from __future__ import annotations

from html import escape
from typing import Any

from faculty_content import (
    ACTIVITY_INFO,
    APP_NAME,
    APP_TAGLINE,
    activity_snapshot,
    describe_result,
    follow_up_suggestions,
    influence_levels,
    reference_rows,
)

PORTAL_NAME = "Faculty Advising Portal"

_SHIELD = "M28 2 52 10v20c0 16-11 27-24 32C15 57 4 46 4 30V10Z"
_BOOK = (
    '<path d="M15 25c4.5-2.2 9-2 13 1 4-3 8.5-3.2 13-1v17c-4.5-2.2-9-2-13 1-4-3-8.5-3.2-13-1Z"/>'
    '<path d="M28 26v17"/>'
)
CREST_LIGHT = (  # white line art for the crimson masthead
    '<svg viewBox="0 0 56 64" width="100%" height="100%" fill="none" stroke="#fff" stroke-width="2.2" '
    f'stroke-linejoin="round" aria-hidden="true"><path d="{_SHIELD}"/>'
    '<path d="M28 7.5 47 14v16c0 13-9 22-19 26.5C18 52 9 43 9 30V14Z" stroke-opacity=".45" stroke-width="1.2"/>'
    f"{_BOOK}</svg>"
)
CREST_SOLID = (  # crimson shield for light backgrounds
    '<svg viewBox="0 0 56 64" width="100%" height="100%" aria-hidden="true">'
    f'<path d="{_SHIELD}" fill="#990000"/>'
    f'<g fill="none" stroke="#fff" stroke-width="2.2" stroke-linejoin="round">{_BOOK}</g></svg>'
)
_ICON_CHECK = (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.6" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>'
)
_ICON_ALERT = (
    '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" '
    'stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5h.01"/></svg>'
)
_ICON_PERSON = (
    '<svg viewBox="0 0 24 24" width="44" height="44" fill="none" stroke="currentColor" stroke-width="1.4" '
    'stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/></svg>'
)


def masthead() -> str:
    return (
        f'<div class="masthead"><div class="crest">{CREST_LIGHT}</div><div>'
        f'<div class="mast-eyebrow">{PORTAL_NAME}</div>'
        f'<div class="mast-title">{APP_NAME}</div>'
        f'<div class="mast-sub">{APP_TAGLINE}</div></div></div>'
    )


def section_header(num: int | str, title: str, subtitle: str = "") -> str:
    sub = f'<div class="sec-sub">{subtitle}</div>' if subtitle else '<div style="height:10px"></div>'
    return f'<div class="sec-h"><span class="sec-n">{num}</span><span class="sec-t">{title}</span></div>{sub}'


def group_header(title: str) -> str:
    return f'<div class="group-h">{title}</div>'


def empty_result() -> str:
    return (
        f'<div class="empty"><div class="ico">{_ICON_PERSON}</div>'
        '<div class="t">No student checked yet</div>'
        '<div class="d">Enter a student&#39;s activity and press <b>Check student</b>,<br>'
        "or load one of the example students.</div></div>"
    )


def result_block(result: dict[str, Any], name: str, course: str) -> str:
    desc = describe_result(result)
    who = f"<b>{escape(name) if name else 'Unnamed student'}</b>"
    if course:
        who += f" · {escape(course)}"
    icon = _ICON_CHECK if desc["on_track"] else _ICON_ALERT
    chance = desc["chance_pct"]
    return (
        f'<div class="{desc["status"]}">'
        f'<div class="result-top"><div class="result-who">{who}</div>'
        f'<span class="pill">{icon}{desc["headline"]}</span></div>'
        f'<div class="chance"><span class="chance-num">{chance}%</span>'
        '<span class="chance-lbl">chance of passing</span></div>'
        f'<div class="why">{desc["explanation"]}</div>'
        f'<div class="meter"><div style="width:{chance}%"></div></div>'
        '<div class="meter-scale"><span>Likely to struggle</span><span>50%</span><span>Likely to pass</span></div>'
        f'<div class="certainty"><b>{desc["certainty"]}.</b> {desc["certainty_detail"]}</div>'
        "</div>"
    )


def next_steps(result: dict[str, Any]) -> str:
    return "".join(
        f'<div class="step"><div class="step-n">{i}</div>'
        f'<div class="step-t"><b>{escape(s["title"])}</b>{escape(s["text"])}</div></div>'
        for i, s in enumerate(follow_up_suggestions(result), start=1)
    )


def comparison(result: dict[str, Any]) -> str:
    return "".join(
        f'<div class="row-item"><div><div class="row-name">{r["label"]}</div>'
        f'<div class="row-meta">This student: <b>{r["value"]}</b> · Typical: {r["typical"]}</div></div>'
        f'<span class="tag {r["status"]}">{r["status_text"]}</span></div>'
        for r in activity_snapshot(result)
    )


def side_guide(feature_stats: dict[str, dict[str, float]]) -> str:
    bench = ["Days.Logged.In", "Submission.Actions", "Create.Actions", "Read.Actions"]
    steps = [
        ("Open your class portal's activity report", "for the student you want to check."),
        ("Type in the numbers", "for each activity, then press Check student."),
        ("Read the result", "and the suggested next steps."),
    ]
    return (
        f'<div class="side-brand"><div class="side-crest">{CREST_SOLID}</div><div>'
        f'<div class="side-name">{APP_NAME}</div><div class="side-sub">{PORTAL_NAME}</div></div></div>'
        '<div class="side-h">How to use</div>'
        + "".join(
            f'<div class="step"><div class="step-n">{i}</div><div class="step-t"><b>{a}</b>{b}</div></div>'
            for i, (a, b) in enumerate(steps, start=1)
        )
        + '<div class="side-h" style="margin-top:22px">A typical passing student</div>'
        + "".join(
            f'<div class="row-item"><span class="row-name">{ACTIVITY_INFO[c]["label"]}</span>'
            f'<b>{feature_stats[c]["high_median"]:.0f}</b></div>'
            for c in bench
        )
        + '<div class="side-note" style="margin-top:22px"><b>Remember:</b> this check is a starting point '
        "for a conversation, not a grade. Weigh it alongside attendance, exam scores, and what you "
        "know about the student.</div>"
    )


def about_page(artifact: dict[str, Any]) -> str:
    summary = artifact["dataset_summary"]
    accuracy_in_10 = round(artifact["metrics"]["accuracy"] * 10)
    cards = [
        (
            "What it does",
            f"It compares a student's online class activity with the records of {summary['total_rows']:,} "
            "past students, and shows whether the pattern looks more like students who passed or "
            "students who struggled.",
        ),
        (
            "When to use it",
            "Early in the term, before midterms, or whenever a student goes quiet, misses deadlines, or "
            "stops logging in. The earlier you check, the more time there is to help.",
        ),
        (
            "How to read the result",
            "Above 50% means the student is on track. Below 50% means they may need support. Results "
            "close to 50% are borderline, so lean more on your own judgement.",
        ),
    ]
    top = "".join(
        f'<div class="panel about-card">{section_header(i, t)}<div class="d">{b}</div></div>'
        for i, (t, b) in enumerate(cards, start=1)
    )
    influence = "".join(
        f'<div class="inf"><div class="inf-name">{r["label"]}</div>'
        f'<div class="inf-bar"><div style="width:{r["bar"]}%"></div></div>'
        f'<span class="lvl {r["level"]}">{r["level"]}</span></div>'
        for r in influence_levels(artifact["feature_importances"])
    )
    ref = (
        '<table class="ref"><thead><tr><th>Activity</th><th class="n">Passed</th><th class="n">Struggled</th>'
        "</tr></thead><tbody>"
        + "".join(
            f'<tr><td><div class="ref-name">{r["Activity"]}</div><div class="ref-help">{r["What it means"]}</div></td>'
            f'<td class="n">{r["Typical passing student"]}</td><td class="n">{r["Typical student who struggled"]}</td></tr>'
            for r in reference_rows(artifact["feature_stats"])
        )
        + "</tbody></table>"
    )
    return (
        f'<div class="grid-3">{top}</div>'
        '<div class="grid-2">'
        f'<div class="panel">{section_header(4, "Which activities matter most?", "How strongly each activity shapes the result.")}{influence}</div>'
        f'<div class="panel">{section_header(5, "Typical activity levels", "What past students usually looked like.")}{ref}</div>'
        "</div>"
        f'<div class="callout"><div><b>Good to know.</b> When tested on past students, this check was right '
        f"about {accuracy_in_10} times out of 10. It only sees online activity, so it can miss things like "
        "exam performance, illness, or work done outside the portal. Use it to decide <i>who to talk to</i>, "
        "never as a final judgement on a student.</div></div>"
    )


def footer(year: int) -> str:
    return f'<div class="app-footer">{PORTAL_NAME} · {APP_NAME} · {year}</div>'
