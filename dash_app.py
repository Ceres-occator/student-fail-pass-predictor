"""Student Success Check: a faculty-facing Dash app (served by gunicorn via `server`)."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html

from faculty_content import (
    ACTIVITY_INFO,
    APP_NAME,
    APP_TAGLINE,
    COURSEWORK_COLS,
    PARTICIPATION_COLS,
    activity_snapshot,
    describe_result,
    follow_up_suggestions,
    influence_levels,
    reference_rows,
    summary_record,
)
from train_model import FEATURE_COLS, get_preset_profiles, load_or_train_artifact, predict_student

artifact = load_or_train_artifact()
presets = get_preset_profiles(artifact)
stats = artifact["feature_stats"]
summary = artifact["dataset_summary"]

app = Dash(__name__, title=APP_NAME)
server = app.server

EXAMPLES = {
    "btn-example-passing": ("passing", "Sample student A"),
    "btn-example-failing": ("failing", "Sample student B"),
    "btn-example-median": ("median", "Sample student C"),
}


def _input_id(col: str) -> str:
    """Dash IDs cannot contain dots."""
    return f"input-{col.replace('.', '-')}"


def _clean_inputs(values) -> dict[str, int]:
    out = {}
    for col, val in zip(FEATURE_COLS, values):
        try:
            out[col] = max(0, int(round(float(val))))
        except (TypeError, ValueError):
            out[col] = 0
    return out


# --------------------------------------------------------------------------- layout
def _field(col: str) -> html.Div:
    info = ACTIVITY_INFO[col]
    return html.Div(
        className="field",
        children=[
            html.Label(
                [info["label"], html.Span("?", className="hint", title=info["help"])],
                htmlFor=_input_id(col),
            ),
            dcc.Input(
                id=_input_id(col),
                type="number",
                min=0,
                step=1,
                value=presets["median"][col],
                className="input",
                debounce=0.6,
            ),
            html.Small(f"Typical passing student: {stats[col]['high_median']:.0f}"),
        ],
    )


def _check_tab() -> html.Div:
    form = html.Div(
        className="card",
        children=[
            html.Div("Student details", className="card-h"),
            html.P(
                "Enter this term's numbers from your class portal. Hover over the ? for a short explanation.",
                className="card-sub",
            ),
            html.Div(
                className="fields",
                children=[
                    html.Div(
                        className="field",
                        children=[
                            html.Label("Student name or ID", htmlFor="student-name"),
                            dcc.Input(id="student-name", type="text", placeholder="Optional",
                                      className="input", debounce=True),
                        ],
                    ),
                    html.Div(
                        className="field",
                        children=[
                            html.Label("Course / section", htmlFor="course-section"),
                            dcc.Input(id="course-section", type="text", placeholder="Optional",
                                      className="input", debounce=True),
                        ],
                    ),
                ],
            ),
            html.Div("Attendance & participation", className="group-h"),
            html.Div(className="fields", children=[_field(c) for c in PARTICIPATION_COLS]),
            html.Div("Coursework", className="group-h"),
            html.Div(className="fields", children=[_field(c) for c in COURSEWORK_COLS]),
            html.Div(
                "The result updates automatically a moment after you change a number.",
                className="live-note",
            ),
        ],
    )
    return html.Div(
        className="check-grid",
        children=[
            html.Div(
                children=[
                    html.P("Trying it for the first time? Load an example student:", className="examples-label"),
                    html.Div(
                        className="examples",
                        children=[
                            html.Button("Doing well", id="btn-example-passing", n_clicks=0, className="btn"),
                            html.Button("Needs support", id="btn-example-failing", n_clicks=0, className="btn"),
                            html.Button("Average", id="btn-example-median", n_clicks=0, className="btn"),
                        ],
                    ),
                    form,
                ]
            ),
            html.Div(
                className="stack",
                children=[
                    html.Div(id="result-panel", className="stack"),
                    html.Button("Download summary", id="btn-download", n_clicks=0, className="btn btn-wide"),
                    dcc.Download(id="download-summary"),
                ],
            ),
        ],
    )


def _about_tab() -> html.Div:
    accuracy_in_10 = round(artifact["metrics"]["accuracy"] * 10)
    cards = [
        (
            "What it does",
            f"It compares a student's online class activity with the records of "
            f"{summary['total_rows']:,} past students, and tells you whether the pattern looks more "
            f"like students who passed or students who struggled.",
        ),
        (
            "When to use it",
            "Early in the term, before midterms, or whenever a student goes quiet, misses deadlines, "
            "or stops logging in. The earlier you check, the more time there is to help.",
        ),
        (
            "How to read the result",
            "Above 50% means the student is on track. Below 50% means they may need support. Results "
            "close to 50% are borderline, so lean more on your own judgement.",
        ),
    ]
    rows = reference_rows(stats)
    return html.Div(
        children=[
            html.Div(
                className="about-grid",
                children=[
                    html.Div(className="card about-card", children=[html.H4(t), html.P(b)])
                    for t, b in cards
                ],
            ),
            html.Div(
                className="card",
                style={"marginBottom": "16px"},
                children=[
                    html.Div("Which activities matter most?", className="card-h"),
                    html.P("How strongly each activity shapes the result, based on past students.",
                           className="card-sub"),
                    *[
                        html.Div(
                            className="inf",
                            children=[
                                html.Div(r["label"], className="inf-name"),
                                html.Div(html.Div(style={"width": f"{r['bar']}%"}), className="inf-bar"),
                                html.Span(r["level"], className=f"lvl {r['level']}"),
                            ],
                        )
                        for r in influence_levels(artifact["feature_importances"])
                    ],
                ],
            ),
            html.Div(
                className="card",
                children=[
                    html.Div("Typical activity levels", className="card-h"),
                    html.P("What past students who passed, and those who struggled, usually looked like.",
                           className="card-sub"),
                    html.Table(
                        className="ref",
                        children=[
                            html.Thead(html.Tr([
                                html.Th("Activity"),
                                html.Th("Passing student", className="n"),
                                html.Th("Student who struggled", className="n"),
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td([
                                        html.Div(r["Activity"], className="ref-name"),
                                        html.Div(r["What it means"], className="ref-help"),
                                    ]),
                                    html.Td(r["Typical passing student"], className="n"),
                                    html.Td(r["Typical student who struggled"], className="n"),
                                ])
                                for r in rows
                            ]),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="callout",
                children=[
                    html.B("Good to know. "),
                    f"When tested on past students, this check was right about {accuracy_in_10} times "
                    "out of 10. It only sees online activity, so it can miss things like exam "
                    "performance, illness, or work outside the portal. Use it to decide who to talk to, "
                    "never as a final judgement on a student.",
                ],
            ),
        ]
    )


app.layout = html.Div(
    className="shell",
    children=[
        html.Header(
            className="page-head",
            children=[
                html.Div(
                    className="brand-mark",
                    children=html.Img(
                        src=app.get_asset_url("cap.svg"), width=28, height=28, alt=""
                    ),
                ),
                html.Div([
                    html.Div("Faculty advising tool", className="eyebrow"),
                    html.H1(APP_NAME),
                    html.P(APP_TAGLINE),
                ]),
            ],
        ),
        dcc.Tabs(
            className="tabs-row",
            parent_className="tabs-parent",
            mobile_breakpoint=0,
            children=[
                dcc.Tab(label="Check a student", className="tab", selected_className="tab--selected",
                        children=_check_tab()),
                dcc.Tab(label="About this tool", className="tab", selected_className="tab--selected",
                        children=_about_tab()),
            ],
        ),
        html.Footer(
            f"{APP_NAME} · For academic advising and student support · {datetime.now().year}",
            className="app-footer",
        ),
    ],
)


# --------------------------------------------------------------------------- callbacks
@app.callback(
    [Output(_input_id(col), "value") for col in FEATURE_COLS]
    + [Output("student-name", "value"), Output("course-section", "value")],
    [Input(btn, "n_clicks") for btn in EXAMPLES],
    prevent_initial_call=True,
)
def load_example(*_clicks):
    """Fill the form with one of the example students."""
    key, name = EXAMPLES.get(ctx.triggered_id, ("median", "Sample student C"))
    profile = presets[key]
    return [profile[col] for col in FEATURE_COLS] + [name, "Sample class"]


def render_result(inputs: dict[str, int], name: str, course: str) -> list:
    result = predict_student(artifact, inputs)
    desc = describe_result(result)
    chance = desc["chance_pct"]
    who = [html.B(name or "Unnamed student")]
    if course:
        who.append(f" · {course}")

    verdict = html.Div(
        className=f"result {desc['status']}",
        children=[
            html.Div(who, className="result-who"),
            html.Span(("✓ " if desc["on_track"] else "! ") + desc["headline"], className="pill"),
            html.H2(f"{chance}% chance of passing"),
            html.P(desc["explanation"], className="why"),
            html.Div(html.Div(style={"width": f"{chance}%"}), className="meter"),
            html.Div([html.Span("Likely to struggle"), html.Span("Likely to pass")], className="meter-scale"),
            html.Div([html.B(desc["certainty"] + ". "), desc["certainty_detail"]], className="certainty"),
        ],
    )
    steps = html.Div(
        className="card",
        children=[html.Div("Suggested next steps", className="list-h")]
        + [
            html.Div(
                className="next",
                children=[
                    html.Div(str(i), className="num-dot"),
                    html.Div([html.B(s["title"]), s["text"]], className="next-t"),
                ],
            )
            for i, s in enumerate(follow_up_suggestions(result), start=1)
        ],
    )
    snapshot = html.Div(
        className="card",
        children=[html.Div("Compared with a typical passing student", className="list-h")]
        + [
            html.Div(
                className="act",
                children=[
                    html.Div([
                        html.Div(r["label"], className="act-name"),
                        html.Div(["This student: ", html.B(r["value"]), f" · Typical: {r['typical']}"],
                                 className="act-num"),
                    ]),
                    html.Span(r["status_text"], className=f"tag {r['status']}"),
                ],
            )
            for r in activity_snapshot(result)
        ],
    )
    return [verdict, steps, snapshot]


@app.callback(
    Output("result-panel", "children"),
    [Input(_input_id(col), "value") for col in FEATURE_COLS]
    + [Input("student-name", "value"), Input("course-section", "value")],
)
def update_prediction(*values):
    """Recalculate the result whenever any field changes."""
    inputs = _clean_inputs(values[: len(FEATURE_COLS)])
    name, course = ((v or "").strip() for v in (list(values[len(FEATURE_COLS):]) + ["", ""])[:2])
    return render_result(inputs, name, course)


@app.callback(
    Output("download-summary", "data"),
    Input("btn-download", "n_clicks"),
    [State(_input_id(col), "value") for col in FEATURE_COLS]
    + [State("student-name", "value"), State("course-section", "value")],
    prevent_initial_call=True,
)
def download_summary(_n, *values):
    inputs = _clean_inputs(values[: len(FEATURE_COLS)])
    name = (values[-2] or "").strip()
    course = (values[-1] or "").strip()
    record = summary_record(name, course, inputs, predict_student(artifact, inputs))
    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "student"
    return dcc.send_data_frame(
        pd.DataFrame([record]).to_csv, f"{safe}_check_{datetime.now():%Y%m%d}.csv", index=False
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
