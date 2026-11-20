"""Student Success Check: faculty-facing Streamlit app (university crimson & white theme)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import ui_html as ui
from faculty_content import ACTIVITY_INFO, APP_NAME, COURSEWORK_COLS, PARTICIPATION_COLS, summary_record
from train_model import FEATURE_COLS, get_preset_profiles, load_or_train_artifact, predict_student

st.set_page_config(
    page_title=APP_NAME,
    page_icon=":material/school:",
    layout="wide",
    initial_sidebar_state="auto",
)

SHARED_CSS = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")

# Streamlit-specific rules that make native widgets match the shared theme.
STREAMLIT_CSS = """
html, body, .stApp, p, label, input, textarea, button, td, th, li {
  font-family: var(--sans) !important;
}
.stApp { background: var(--bg); }
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1240px; }
.stMarkdown p { margin-bottom: 0; }

/* Cards: bordered containers and the form share one look */
div[class*="st-key-card-"], div[data-testid="stForm"] {
  background: var(--surface); border: 1px solid var(--line) !important; border-top: 3px solid var(--brand) !important;
  border-radius: var(--radius) !important;
}
div[class*="st-key-card-"], div[data-testid="stForm"] { padding: 20px 22px 22px !important; }

/* Sidebar */
section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--line); margin-bottom: 6px; }
.stTabs [data-baseweb="tab"] { padding: 10px 18px; font-weight: 600; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--brand) !important; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--brand); height: 3px; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Buttons */
div.stButton > button {
  border-radius: 8px; border: 1px solid var(--line); background: var(--surface); color: var(--ink);
  font-weight: 500; min-height: 42px; transition: border-color .15s, background .15s, color .15s;
}
div.stButton > button:hover, div.stButton > button:focus:not(:active) {
  border-color: var(--brand); background: var(--brand-soft); color: var(--brand);
}
div[data-testid="stFormSubmitButton"] > button {
  background: var(--brand); color: #fff; border: none; border-radius: 8px; font-weight: 600; min-height: 46px;
}
div[data-testid="stFormSubmitButton"] > button:hover { background: var(--brand-dark); color: #fff; }
div[data-testid="stDownloadButton"] > button {
  border-radius: 8px; border: 1.5px solid var(--brand); color: var(--brand); font-weight: 600;
  background: var(--surface); min-height: 44px;
}
div[data-testid="stDownloadButton"] > button:hover { background: var(--brand); color: #fff; }

/* Inputs */
div[data-testid="stWidgetLabel"] p { font-size: 0.86rem; font-weight: 500; color: var(--text); }
div[data-testid="stNumberInput"] input, div[data-testid="stTextInput"] input { font-weight: 500; color: var(--ink); }
"""


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Getting things ready…")
def get_artifact() -> dict:
    return load_or_train_artifact()


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


def render_inputs(artifact: dict, presets: dict[str, dict[str, int]]) -> None:
    stats = artifact["feature_stats"]

    with st.container(border=True, key="card-examples"):
        html(ui.section_header(1, "Start with an example", "Optional: load a sample student to see how it works."))
        b1, b2, b3 = st.columns(3)
        examples = [
            (b1, "Doing well", "passing", "Sample student A"),
            (b2, "Needs support", "failing", "Sample student B"),
            (b3, "Average", "median", "Sample student C"),
        ]
        for box, label, key, name in examples:
            box.button(label, width="stretch", on_click=_load_example, args=(presets[key], name))

    with st.form("student_form", border=True):
        html(ui.section_header(2, "Student details", "Enter this term's numbers from your class portal. "
                                                     "Hover over the ? next to each item for an explanation."))
        n1, n2 = st.columns(2)
        n1.text_input("Student name or ID", key="student_name", placeholder="Optional")
        n2.text_input("Course / section", key="course_section", placeholder="Optional")

        for title, cols in (("Attendance & participation", PARTICIPATION_COLS), ("Coursework", COURSEWORK_COLS)):
            html(ui.group_header(title))
            for start in range(0, len(cols), 2):
                for box, col in zip(st.columns(2), cols[start : start + 2]):
                    info = ACTIVITY_INFO[col]
                    box.number_input(
                        info["label"],
                        min_value=0,
                        step=1,
                        key=f"input_{col}",
                        help=f"{info['help']}  \nTypical passing student: **{stats[col]['high_median']:.0f}**",
                    )

        st.write("")
        if st.form_submit_button("Check student", icon=":material/search:", type="primary", width="stretch"):
            st.session_state["show_result"] = True


def render_result(artifact: dict) -> None:
    with st.container(border=True, key="card-result"):
        html(ui.section_header(3, "Result"))
        if not st.session_state.get("show_result"):
            html(ui.empty_result())
            return
        inputs = {col: int(st.session_state[f"input_{col}"] or 0) for col in FEATURE_COLS}
        result = predict_student(artifact, inputs)
        name = st.session_state.get("student_name", "").strip()
        course = st.session_state.get("course_section", "").strip()
        html(ui.result_block(result, name, course))

    with st.container(border=True, key="card-steps"):
        html(ui.section_header(4, "Suggested next steps"))
        html(ui.next_steps(result))

    with st.container(border=True, key="card-compare"):
        html(ui.section_header(5, "Compared with a typical passing student"))
        html(ui.comparison(result))

    safe = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "student"
    st.download_button(
        "Download summary",
        icon=":material/download:",
        data=pd.DataFrame([summary_record(name, course, inputs, result)]).to_csv(index=False).encode("utf-8"),
        file_name=f"{safe}_check_{datetime.now():%Y%m%d}.csv",
        mime="text/csv",
        width="stretch",
        help="Saves a spreadsheet file (opens in Excel or Google Sheets) for your records.",
    )


def main() -> None:
    html(f"<style>{SHARED_CSS}\n{STREAMLIT_CSS}</style>")
    artifact = get_artifact()
    presets = get_preset_profiles(artifact)
    _init_state(presets)

    with st.sidebar:
        html(ui.side_guide(artifact["feature_stats"]))

    html(ui.masthead())
    tab_check, tab_about = st.tabs([":material/person_search: Check a student", ":material/info: About this tool"])
    with tab_check:
        left, right = st.columns(2, gap="large")
        with left:
            render_inputs(artifact, presets)
        with right:
            render_result(artifact)
    with tab_about:
        html(ui.about_page(artifact))

    html(ui.footer(datetime.now().year))


if __name__ == "__main__":
    main()
