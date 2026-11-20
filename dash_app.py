"""Student Success Check: faculty-facing Dash app (served by gunicorn via `server`)."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from dash import Dash, Input, Output, State, ctx, dcc, html

import ui_html as ui
from faculty_content import ACTIVITY_INFO, APP_NAME, COURSEWORK_COLS, PARTICIPATION_COLS, summary_record
from train_model import FEATURE_COLS, get_preset_profiles, load_or_train_artifact, predict_student

artifact = load_or_train_artifact()
presets = get_preset_profiles(artifact)
stats = artifact["feature_stats"]

app = Dash(__name__, title=APP_NAME)
server = app.server

EXAMPLES = {
    "btn-example-passing": ("passing", "Sample student A"),
    "btn-example-failing": ("failing", "Sample student B"),
    "btn-example-median": ("median", "Sample student C"),
}


def _md(markup: str, **kwargs) -> dcc.Markdown:
    """Render one of the shared HTML blocks."""
    return dcc.Markdown(markup, dangerously_allow_html=True, className="md", **kwargs)


def _input_id(col: str) -> str:
    return f"input-{col.replace('.', '-')}"  # Dash IDs cannot contain dots


def _clean_inputs(values) -> dict[str, int]:
    out = {}
    for col, val in zip(FEATURE_COLS, values):
        try:
            out[col] = max(0, int(round(float(val))))
        except (TypeError, ValueError):
            out[col] = 0
    return out


def _field(col: str) -> html.Div:
    info = ACTIVITY_INFO[col]
    return html.Div(
        className="field",
        children=[
            html.Label([info["label"], html.Span("?", className="hint", title=info["help"])], htmlFor=_input_id(col)),
            dcc.Input(id=_input_id(col), type="number", min=0, step=1, value=presets["median"][col],
                      className="input", debounce=0.6),
            html.Small(f"Typical passing student: {stats[col]['high_median']:.0f}"),
        ],
    )


def _text_field(label: str, id_: str) -> html.Div:
    return html.Div(
        className="field",
        children=[html.Label(label, htmlFor=id_),
                  dcc.Input(id=id_, type="text", placeholder="Optional", className="input", debounce=True)],
    )


def _check_tab() -> html.Div:
    examples = html.Div(
        className="panel",
        children=[
            _md(ui.section_header(1, "Start with an example", "Optional: load a sample student to see how it works.")),
            html.Div(
                className="examples",
                children=[
                    html.Button("Doing well", id="btn-example-passing", n_clicks=0, className="btn"),
                    html.Button("Needs support", id="btn-example-failing", n_clicks=0, className="btn"),
                    html.Button("Average", id="btn-example-median", n_clicks=0, className="btn"),
                ],
            ),
        ],
    )
    form = html.Div(
        className="panel",
        children=[
            _md(ui.section_header(2, "Student details", "Enter this term's numbers from your class portal. "
                                                       "Hover over the ? next to each item for an explanation.")),
            html.Div(className="fields", children=[_text_field("Student name or ID", "student-name"),
                                                   _text_field("Course / section", "course-section")]),
            _md(ui.group_header("Attendance &amp; participation")),
            html.Div(className="fields", children=[_field(c) for c in PARTICIPATION_COLS]),
            _md(ui.group_header("Coursework")),
            html.Div(className="fields", children=[_field(c) for c in COURSEWORK_COLS]),
            html.Div("The result updates automatically a moment after you change a number.", className="live-note"),
        ],
    )
    return html.Div(
        className="two-col",
        children=[
            html.Div(className="col", children=[examples, form]),
            html.Div(
                className="col",
                children=[
                    html.Div(id="result-panel", className="col"),
                    html.Button("Download summary", id="btn-download", n_clicks=0, className="btn btn-outline"),
                    dcc.Download(id="download-summary"),
                ],
            ),
        ],
    )


app.layout = html.Div(
    className="shell",
    children=[
        _md(ui.masthead()),
        dcc.Tabs(
            className="tabs-row",
            mobile_breakpoint=0,
            children=[
                dcc.Tab(label="Check a student", className="tab", selected_className="tab--selected",
                        children=_check_tab()),
                dcc.Tab(label="About this tool", className="tab", selected_className="tab--selected",
                        children=_md(ui.about_page(artifact))),
            ],
        ),
        _md(ui.footer(datetime.now().year)),
    ],
)


@app.callback(
    [Output(_input_id(col), "value") for col in FEATURE_COLS]
    + [Output("student-name", "value"), Output("course-section", "value")],
    [Input(btn, "n_clicks") for btn in EXAMPLES],
    prevent_initial_call=True,
)
def load_example(*_clicks):
    """Fill the form with one of the example students."""
    key, name = EXAMPLES.get(ctx.triggered_id, ("median", "Sample student C"))
    return [presets[key][col] for col in FEATURE_COLS] + [name, "Sample class"]


@app.callback(
    Output("result-panel", "children"),
    [Input(_input_id(col), "value") for col in FEATURE_COLS]
    + [Input("student-name", "value"), Input("course-section", "value")],
)
def update_prediction(*values):
    """Recalculate the result whenever any field changes."""
    inputs = _clean_inputs(values[: len(FEATURE_COLS)])
    extra = list(values[len(FEATURE_COLS):]) + [None, None]
    name, course = ((v or "").strip() for v in extra[:2])
    result = predict_student(artifact, inputs)
    return [
        html.Div(className="panel", children=_md(ui.section_header(3, "Result") + ui.result_block(result, name, course))),
        html.Div(className="panel", children=_md(ui.section_header(4, "Suggested next steps") + ui.next_steps(result))),
        html.Div(className="panel", children=_md(
            ui.section_header(5, "Compared with a typical passing student") + ui.comparison(result))),
    ]


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
    return dcc.send_data_frame(pd.DataFrame([record]).to_csv, f"{safe}_check_{datetime.now():%Y%m%d}.csv", index=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
