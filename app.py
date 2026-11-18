"""Student Success Check: a faculty-facing Streamlit app."""

from __future__ import annotations

from datetime import datetime
from html import escape

import pandas as pd
import streamlit as st

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

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="auto",
)

# --------------------------------------------------------------------------- icons
ICON_CAP = (
    '<svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M22 10 12 5 2 10l10 5 10-5Z"/><path d="M6 12v5c3 2 9 2 12 0v-5"/>'
    '<path d="M22 10v6"/></svg>'
)
ICON_CHECK = (
    '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" '
    'stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>'
)
ICON_ALERT = (
    '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>'
    '<path d="M12 8v5"/><path d="M12 16.5h.01"/></svg>'
)
ICON_INFO = (
    '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="9"/>'
    '<path d="M12 11v5"/><path d="M12 7.5h.01"/></svg>'
)
ICON_PERSON = (
    '<svg viewBox="0 0 24 24" width="40" height="40" fill="none" stroke="currentColor" '
    'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="4"/>'
    '<path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/></svg>'
)

# --------------------------------------------------------------------------- styles
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --brand: #1e3a64;
  --brand-2: #2c5491;
  --brand-soft: #eef3fb;
  --bg: #f5f7fb;
  --surface: #ffffff;
  --line: #e3e8f0;
  --ink: #0f1d33;
  --text: #334155;
  --muted: #64748b;
  --good: #157347;
  --good-bg: #ecf8f1;
  --good-line: #bfe6cf;
  --warn: #b45309;
  --warn-bg: #fff6eb;
  --warn-line: #f6d4a8;
  --radius: 14px;
}

html, body, .stApp, .stMarkdown, .stMarkdown div, p, label, input, textarea, button, td, th, li, h1, h2, h3, h4 {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}
.stApp { background: var(--bg); }
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1280px; }

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
.brand { display: flex; align-items: center; gap: 10px; margin: 2px 0 22px 0; }
.brand-mark {
  width: 38px; height: 38px; border-radius: 10px; background: var(--brand); color: #fff;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.brand-name { font-weight: 700; color: var(--ink); font-size: 1rem; line-height: 1.2; }
.brand-sub { font-size: 0.78rem; color: var(--muted); }
.side-h { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted);
  font-weight: 600; margin: 4px 0 12px 0; }
.step { display: flex; gap: 12px; margin-bottom: 14px; }
.step-n { flex: 0 0 24px; height: 24px; border-radius: 50%; background: var(--brand-soft); color: var(--brand);
  font-size: 0.78rem; font-weight: 700; display: flex; align-items: center; justify-content: center; }
.step-t { font-size: 0.88rem; color: var(--text); line-height: 1.45; }
.step-t b { color: var(--ink); font-weight: 600; }
.bench { display: flex; justify-content: space-between; gap: 8px; padding: 8px 0;
  border-bottom: 1px solid var(--line); font-size: 0.87rem; color: var(--text); }
.bench:last-child { border-bottom: none; }
.bench b { color: var(--ink); font-weight: 600; }
.side-note { background: var(--brand-soft); border-radius: 10px; padding: 12px 14px; font-size: 0.82rem;
  color: var(--text); line-height: 1.5; }

/* ---------- Header ---------- */
.page-head {
  background: linear-gradient(115deg, var(--brand) 0%, var(--brand-2) 100%);
  border-radius: 18px; padding: 26px 30px; color: #fff; margin-bottom: 18px;
  box-shadow: 0 10px 28px rgba(30, 58, 100, 0.18);
}
.page-head .eyebrow { font-size: 0.75rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: #c9d8f0; font-weight: 600; }
.page-head h1 { color: #fff !important; font-size: 1.85rem !important; font-weight: 700 !important;
  margin: 4px 0 6px 0 !important; padding: 0 !important; letter-spacing: -0.01em; }
.page-head p { color: #dbe5f5; margin: 0; font-size: 0.98rem; max-width: 720px; line-height: 1.55; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 10px 16px; font-weight: 600; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--brand) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--brand); height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- Cards ---------- */
div[data-testid="stForm"], div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface); border: 1px solid var(--line) !important; border-radius: var(--radius);
  box-shadow: 0 1px 2px rgba(15, 29, 51, 0.04);
}
div[data-testid="stForm"] { padding: 20px 22px 22px 22px; }
.card-h { font-size: 1.02rem; font-weight: 700; color: var(--ink); margin: 0 0 2px 0; }
.card-sub { font-size: 0.86rem; color: var(--muted); margin: 0 0 14px 0; line-height: 1.5; }
.group-h { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; font-weight: 600;
  color: var(--brand); margin: 16px 0 6px 0; padding-top: 14px; border-top: 1px solid var(--line); }
.examples-label { font-size: 0.86rem; color: var(--muted); margin: 0 0 8px 0; }

/* ---------- Buttons & inputs ---------- */
div.stButton > button {
  border-radius: 10px; border: 1px solid var(--line); background: var(--surface); color: var(--ink);
  font-weight: 500; padding: 0.5rem 0.8rem; transition: border-color .15s, background .15s;
}
div.stButton > button:hover { border-color: var(--brand-2); background: var(--brand-soft); color: var(--brand); }
div[data-testid="stFormSubmitButton"] > button {
  background: var(--brand); color: #fff; border: none; border-radius: 10px; font-weight: 600;
  padding: 0.7rem 1rem; font-size: 0.98rem;
}
div[data-testid="stFormSubmitButton"] > button:hover { background: var(--brand-2); color: #fff; }
div[data-testid="stDownloadButton"] > button {
  border-radius: 10px; border: 1px solid var(--line); color: var(--brand); font-weight: 600; background: var(--surface);
}
div[data-testid="stDownloadButton"] > button:hover { border-color: var(--brand-2); background: var(--brand-soft); }
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input { font-weight: 500; color: var(--ink); }
div[data-testid="stWidgetLabel"] p { font-size: 0.86rem; font-weight: 500; color: var(--text); }

/* ---------- Result ---------- */
.result { border-radius: var(--radius); padding: 22px 22px 20px 22px; border: 1px solid; }
.result.on-track { background: var(--good-bg); border-color: var(--good-line); }
.result.needs-support { background: var(--warn-bg); border-color: var(--warn-line); }
.result-who { font-size: 0.82rem; color: var(--muted); margin-bottom: 12px; }
.result-who b { color: var(--ink); font-weight: 600; font-size: 0.95rem; }
.pill { display: inline-flex; align-items: center; gap: 6px; border-radius: 999px; padding: 5px 12px;
  font-weight: 600; font-size: 0.82rem; color: #fff; }
.on-track .pill { background: var(--good); }
.needs-support .pill { background: var(--warn); }
.result h2 { font-size: 1.55rem !important; font-weight: 700 !important; color: var(--ink) !important;
  margin: 12px 0 6px 0 !important; padding: 0 !important; letter-spacing: -0.01em; }
.result .why { color: var(--text); font-size: 0.93rem; line-height: 1.55; margin: 0 0 18px 0; }
.meter-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.meter-row span { font-size: 0.84rem; color: var(--muted); }
.meter-row strong { font-size: 1.35rem; color: var(--ink); font-weight: 700; }
.meter { position: relative; height: 10px; border-radius: 99px; background: rgba(15, 29, 51, 0.08); }
.meter > div { height: 100%; border-radius: 99px; }
.on-track .meter > div { background: var(--good); }
.needs-support .meter > div { background: var(--warn); }
.meter::after { content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 2px;
  background: rgba(15, 29, 51, 0.35); border-radius: 2px; }
.meter-scale { display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted); margin-top: 6px; }
.certainty { margin-top: 12px; font-size: 0.84rem; color: var(--text); }
.certainty b { font-weight: 600; color: var(--ink); }

.list-h { font-size: 0.95rem; font-weight: 700; color: var(--ink); margin: 0 0 10px 0; }
.act { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 9px 0;
  border-bottom: 1px solid var(--line); }
.act:last-child { border-bottom: none; }
.act-name { font-size: 0.88rem; color: var(--ink); font-weight: 500; }
.act-num { font-size: 0.78rem; color: var(--muted); margin-top: 1px; }
.tag { flex-shrink: 0; font-size: 0.74rem; font-weight: 600; padding: 3px 9px; border-radius: 999px; }
.tag.good { background: var(--good-bg); color: var(--good); }
.tag.low { background: var(--warn-bg); color: var(--warn); }
.tag.info { background: #f1f4f9; color: var(--muted); }

.next { display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid var(--line); }
.next:last-child { border-bottom: none; }
.next-n { flex: 0 0 24px; height: 24px; border-radius: 50%; background: var(--brand-soft); color: var(--brand);
  font-size: 0.78rem; font-weight: 700; display: flex; align-items: center; justify-content: center; }
.next-t { font-size: 0.88rem; color: var(--text); line-height: 1.5; }
.next-t b { color: var(--ink); font-weight: 600; display: block; }

.empty { text-align: center; padding: 48px 24px; border: 1.5px dashed #cdd6e4; border-radius: var(--radius);
  background: var(--surface); color: var(--muted); }
.empty .ico { color: #a9b6ca; margin-bottom: 8px; }
.empty h4 { color: var(--ink) !important; font-size: 1rem !important; font-weight: 600 !important;
  margin: 0 0 6px 0 !important; padding: 0 !important; }
.empty p { font-size: 0.88rem; margin: 0; line-height: 1.55; }

/* ---------- About tab ---------- */
.about-card { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 18px 20px; height: 100%; }
.about-card h4 { margin: 0 0 6px 0 !important; padding: 0 !important; color: var(--ink) !important;
  font-size: 0.98rem !important; font-weight: 700 !important; }
.about-card p { margin: 0; color: var(--text); font-size: 0.89rem; line-height: 1.55; }
.inf { display: grid; grid-template-columns: minmax(150px, 1.1fr) 2fr auto; align-items: center; gap: 14px;
  padding: 10px 0; border-bottom: 1px solid var(--line); }
.inf:last-child { border-bottom: none; }
.inf-name { font-size: 0.9rem; color: var(--ink); font-weight: 500; }
.inf-bar { height: 8px; border-radius: 99px; background: #eef1f6; overflow: hidden; }
.inf-bar div { height: 100%; border-radius: 99px; background: var(--brand-2); }
.lvl { font-size: 0.74rem; font-weight: 600; padding: 3px 10px; border-radius: 999px; min-width: 72px; text-align: center; }
.lvl.Strong { background: var(--brand); color: #fff; }
.lvl.Moderate { background: var(--brand-soft); color: var(--brand); }
.lvl.Minor { background: #f1f4f9; color: var(--muted); }
.ref { width: 100%; border-collapse: collapse; margin-top: 4px; }
.ref th { text-align: left; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); font-weight: 600; padding: 8px 10px; border-bottom: 1px solid var(--line); background: transparent; }
.ref td { padding: 10px; border-bottom: 1px solid var(--line); vertical-align: top; border-left: none; border-right: none; }
.ref tr:last-child td { border-bottom: none; }
.ref .n { text-align: right; white-space: nowrap; width: 1%; padding-left: 24px; }
.ref td.n { font-weight: 600; color: var(--ink); font-size: 0.95rem; }
.ref-name { font-size: 0.9rem; color: var(--ink); font-weight: 500; }
.ref-help { font-size: 0.8rem; color: var(--muted); margin-top: 2px; line-height: 1.45; }
.callout { display: flex; gap: 12px; background: var(--brand-soft); border-radius: var(--radius);
  padding: 16px 18px; color: var(--text); font-size: 0.9rem; line-height: 1.55; }
.callout .ico { color: var(--brand); flex-shrink: 0; margin-top: 1px; }
.callout b { color: var(--ink); }

.app-footer { margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--line); text-align: center;
  color: var(--muted); font-size: 0.8rem; }

@media (max-width: 640px) {
  .page-head { padding: 20px; }
  .page-head h1 { font-size: 1.45rem !important; }
  .inf { grid-template-columns: 1fr auto; }
  .inf-bar { display: none; }
}
</style>
"""


def html(markup: str) -> None:
    """Render trusted HTML (lines stripped so Markdown never treats indentation as code)."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


@st.cache_resource(show_spinner="Getting things ready…")
def get_artifact() -> dict:
    return load_or_train_artifact()


# --------------------------------------------------------------------------- state
def _load_example(profile: dict[str, int], name: str) -> None:
    for col, value in profile.items():
        st.session_state[f"input_{col}"] = int(value)
    st.session_state["student_name"] = name
    st.session_state["course_section"] = "Sample class"
    st.session_state["show_result"] = True


def _init_state(presets: dict[str, dict[str, int]]) -> None:
    for col in FEATURE_COLS:
        st.session_state.setdefault(f"input_{col}", int(presets["median"][col]))
    st.session_state.setdefault("student_name", "")
    st.session_state.setdefault("course_section", "")
    st.session_state.setdefault("show_result", False)


# --------------------------------------------------------------------------- layout pieces
def render_sidebar(artifact: dict) -> None:
    stats = artifact["feature_stats"]
    with st.sidebar:
        html(
            f"""
            <div class="brand">
              <div class="brand-mark">{ICON_CAP}</div>
              <div><div class="brand-name">{APP_NAME}</div>
              <div class="brand-sub">Early support for your students</div></div>
            </div>
            <div class="side-h">How to use</div>
            <div class="step"><div class="step-n">1</div><div class="step-t">
              <b>Open your class portal's activity report</b> for the student.</div></div>
            <div class="step"><div class="step-n">2</div><div class="step-t">
              <b>Type in the numbers</b> for each activity, then press <b>Check student</b>.</div></div>
            <div class="step"><div class="step-n">3</div><div class="step-t">
              <b>Read the result</b> and the suggested next steps.</div></div>
            """
        )
        st.divider()
        html('<div class="side-h">A typical passing student</div>')
        bench_cols = ["Days.Logged.In", "Submission.Actions", "Create.Actions", "Read.Actions"]
        html(
            "".join(
                f'<div class="bench"><span>{ACTIVITY_INFO[c]["label"]}</span>'
                f'<b>{stats[c]["high_median"]:.0f}</b></div>'
                for c in bench_cols
            )
        )
        st.divider()
        html(
            """
            <div class="side-note"><b>Remember:</b> this check is a starting point for a
            conversation, not a grade. Always weigh it alongside attendance, exam scores,
            and what you know about the student.</div>
            """
        )


def render_header() -> None:
    html(
        f"""
        <div class="page-head">
          <div class="eyebrow">Faculty advising tool</div>
          <h1>{APP_NAME}</h1>
          <p>{APP_TAGLINE}</p>
        </div>
        """
    )


def render_form(artifact: dict, presets: dict[str, dict[str, int]]) -> None:
    stats = artifact["feature_stats"]

    html('<p class="examples-label">Trying it for the first time? Load an example student:</p>')
    b1, b2, b3 = st.columns(3)
    b1.button(
        "Doing well",
        icon=":material/sentiment_satisfied:",
        width="stretch",
        on_click=_load_example,
        args=(presets["passing"], "Sample student A"),
    )
    b2.button(
        "Needs support",
        icon=":material/support:",
        width="stretch",
        on_click=_load_example,
        args=(presets["failing"], "Sample student B"),
    )
    b3.button(
        "Average",
        icon=":material/person:",
        width="stretch",
        on_click=_load_example,
        args=(presets["median"], "Sample student C"),
    )

    with st.form("student_form", border=True):
        html(
            """
            <div class="card-h">Student details</div>
            <div class="card-sub">Enter this term's numbers from your class portal.
            Hover over the <b>?</b> next to each item for a short explanation.</div>
            """
        )
        n1, n2 = st.columns(2)
        n1.text_input("Student name or ID", key="student_name", placeholder="Optional")
        n2.text_input("Course / section", key="course_section", placeholder="Optional")

        def field(col: str) -> None:
            info = ACTIVITY_INFO[col]
            st.number_input(
                info["label"],
                min_value=0,
                step=1,
                key=f"input_{col}",
                help=f"{info['help']}  \nTypical passing student: **{stats[col]['high_median']:.0f}**",
            )

        def group(title: str, cols: list[str]) -> None:
            html(f'<div class="group-h">{title}</div>')
            for start in range(0, len(cols), 2):
                row = st.columns(2)
                for box, col in zip(row, cols[start : start + 2]):
                    with box:
                        field(col)

        group("Attendance &amp; participation", PARTICIPATION_COLS)
        group("Coursework", COURSEWORK_COLS)

        st.write("")
        if st.form_submit_button("Check student", icon=":material/search:", type="primary", width="stretch"):
            st.session_state["show_result"] = True


def render_result(artifact: dict) -> None:
    if not st.session_state.get("show_result"):
        html(
            f"""
            <div class="empty">
              <div class="ico">{ICON_PERSON}</div>
              <h4>No student checked yet</h4>
              <p>Fill in the student's activity and press <b>Check student</b>,<br>
              or load one of the example students to see how it works.</p>
            </div>
            """
        )
        return

    inputs = {col: int(st.session_state[f"input_{col}"] or 0) for col in FEATURE_COLS}
    result = predict_student(artifact, inputs)
    desc = describe_result(result)
    name = st.session_state.get("student_name", "").strip()
    course = st.session_state.get("course_section", "").strip()

    who = escape(name) if name else "Unnamed student"
    where = f" · {escape(course)}" if course else ""
    icon = ICON_CHECK if desc["on_track"] else ICON_ALERT
    chance = desc["chance_pct"]

    html(
        f"""
        <div class="result {desc['status']}">
          <div class="result-who"><b>{who}</b>{where}</div>
          <span class="pill">{icon}{desc['headline']}</span>
          <h2>{chance}% chance of passing</h2>
          <p class="why">{desc['explanation']}</p>
          <div class="meter"><div style="width:{chance}%"></div></div>
          <div class="meter-scale"><span>Likely to struggle</span><span>Likely to pass</span></div>
          <div class="certainty"><b>{desc['certainty']}.</b> {desc['certainty_detail']}</div>
        </div>
        """
    )

    st.write("")
    with st.container(border=True):
        steps = follow_up_suggestions(result)
        html(
            '<div class="list-h">Suggested next steps</div>'
            + "".join(
                f'<div class="next"><div class="next-n">{i}</div>'
                f'<div class="next-t"><b>{escape(s["title"])}</b>{escape(s["text"])}</div></div>'
                for i, s in enumerate(steps, start=1)
            )
        )

    with st.container(border=True):
        rows = activity_snapshot(result)
        html(
            '<div class="list-h">Compared with a typical passing student</div>'
            + "".join(
                f'<div class="act"><div><div class="act-name">{r["label"]}</div>'
                f'<div class="act-num">This student: <b>{r["value"]}</b> · Typical: {r["typical"]}</div></div>'
                f'<span class="tag {r["status"]}">{r["status_text"]}</span></div>'
                for r in rows
            )
        )

    record = summary_record(name, course, inputs, result)
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "student"
    st.download_button(
        "Download summary",
        icon=":material/download:",
        data=pd.DataFrame([record]).to_csv(index=False).encode("utf-8"),
        file_name=f"{safe_name}_check_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
        width="stretch",
        help="Saves a spreadsheet file (opens in Excel or Google Sheets) for your records.",
    )


def render_about(artifact: dict) -> None:
    summary = artifact["dataset_summary"]
    accuracy_in_10 = round(artifact["metrics"]["accuracy"] * 10)

    cards = [
        (
            "What it does",
            f"It compares a student's online class activity with the records of "
            f"{summary['total_rows']:,} past students, and tells you whether the pattern looks "
            f"more like students who passed or students who struggled.",
        ),
        (
            "When to use it",
            "Early in the term, before midterms, or whenever a student goes quiet, misses "
            "deadlines, or stops logging in. The earlier you check, the more time there is to help.",
        ),
        (
            "How to read the result",
            "Above 50% means the student is on track. Below 50% means they may need support. "
            "Results close to 50% are borderline, so lean more on your own judgement.",
        ),
    ]
    cols = st.columns(3, gap="medium")
    for col, (title, body) in zip(cols, cards):
        with col:
            html(f'<div class="about-card"><h4>{title}</h4><p>{body}</p></div>')

    st.write("")
    with st.container(border=True):
        html(
            '<div class="card-h">Which activities matter most?</div>'
            '<div class="card-sub">How strongly each activity shapes the result, based on past students.</div>'
            + "".join(
                f'<div class="inf"><div class="inf-name">{r["label"]}</div>'
                f'<div class="inf-bar"><div style="width:{r["bar"]}%"></div></div>'
                f'<span class="lvl {r["level"]}">{r["level"]}</span></div>'
                for r in influence_levels(artifact["feature_importances"])
            )
        )

    with st.container(border=True):
        html(
            '<div class="card-h">Typical activity levels</div>'
            '<div class="card-sub">What past students who passed, and those who struggled, usually looked like.</div>'
        )
        rows = reference_rows(artifact["feature_stats"])
        html(
            '<table class="ref"><thead><tr><th>Activity</th><th class="n">Passing student</th>'
            '<th class="n">Student who struggled</th></tr></thead><tbody>'
            + "".join(
                f'<tr><td><div class="ref-name">{r["Activity"]}</div>'
                f'<div class="ref-help">{r["What it means"]}</div></td>'
                f'<td class="n">{r["Typical passing student"]}</td>'
                f'<td class="n">{r["Typical student who struggled"]}</td></tr>'
                for r in rows
            )
            + "</tbody></table>"
        )

    html(
        f"""
        <div class="callout"><div class="ico">{ICON_INFO}</div><div>
        <b>Good to know.</b> When tested on past students, this check was right about
        {accuracy_in_10} times out of 10. It only sees online activity, so it can miss
        things like exam performance, illness, or work outside the portal. Use it to decide
        <i>who to talk to</i>, never as a final judgement on a student.
        </div></div>
        """
    )


def main() -> None:
    html(CSS)
    artifact = get_artifact()
    presets = get_preset_profiles(artifact)
    _init_state(presets)

    render_sidebar(artifact)
    render_header()

    tab_check, tab_about = st.tabs([":material/person_search: Check a student", ":material/info: About this tool"])
    with tab_check:
        left, right = st.columns([1.15, 1], gap="large")
        with left:
            render_form(artifact, presets)
        with right:
            render_result(artifact)
    with tab_about:
        render_about(artifact)

    html(
        f'<div class="app-footer">{APP_NAME} · For academic advising and student support · '
        f"{datetime.now().year}</div>"
    )


if __name__ == "__main__":
    main()
