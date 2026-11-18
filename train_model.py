"""Model training, evaluation, and inference utilities for Student Pass/Fail Prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "augmented_balanced_dataset_m.csv"
ARTIFACT_PATH = BASE_DIR / "model_artifact.joblib"

FEATURE_COLS: list[str] = [
    "Submission.Actions",
    "Quiz.Actions",
    "Forum.Actions",
    "Create.Actions",
    "Read.Actions",
    "Update.Actions",
    "Delete.Actions",
    "Days.Logged.In",
]

FEATURE_META: dict[str, dict[str, str]] = {
    "Submission.Actions": {
        "label": "Submission Actions",
        "help": "Total assignment or coursework submission actions recorded in the LMS.",
    },
    "Quiz.Actions": {
        "label": "Quiz Actions",
        "help": "Total interactions with online quizzes and self-assessments.",
    },
    "Forum.Actions": {
        "label": "Forum Actions",
        "help": "Total posts, replies, and views in course discussion forums.",
    },
    "Create.Actions": {
        "label": "Create Actions",
        "help": "Number of content or entry creation events performed by the student.",
    },
    "Read.Actions": {
        "label": "Read Actions",
        "help": "Total course materials, pages, and resources viewed/read.",
    },
    "Update.Actions": {
        "label": "Update Actions",
        "help": "Number of edits or updates to submissions, profiles, or quiz attempts.",
    },
    "Delete.Actions": {
        "label": "Delete Actions",
        "help": "Number of deletion events recorded in the LMS.",
    },
    "Days.Logged.In": {
        "label": "Days Logged In",
        "help": "Number of distinct days the student logged into the learning platform.",
    },
}

SAMPLE_PRESETS: dict[str, dict[str, int]] = {
    "passing": {
        "Submission.Actions": 12,
        "Quiz.Actions": 0,
        "Forum.Actions": 0,
        "Create.Actions": 16,
        "Read.Actions": 72,
        "Update.Actions": 4,
        "Delete.Actions": 0,
        "Days.Logged.In": 11,
    },
    "failing": {
        "Submission.Actions": 2,
        "Quiz.Actions": 34,
        "Forum.Actions": 0,
        "Create.Actions": 4,
        "Read.Actions": 207,
        "Update.Actions": 3,
        "Delete.Actions": 0,
        "Days.Logged.In": 13,
    },
}


def get_preset_profiles(artifact: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Return preset input profiles including the dataset median student profile."""
    stats = artifact["feature_stats"]
    median_profile = {col: int(round(stats[col]["median"])) for col in FEATURE_COLS}
    return {
        "passing": dict(SAMPLE_PRESETS["passing"]),
        "failing": dict(SAMPLE_PRESETS["failing"]),
        "median": median_profile,
    }


def _evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Compute binary classification metrics and confusion matrix on holdout test data."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def _compute_feature_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute overall and class-stratified descriptive statistics per feature."""
    high_df = df[df["rating"] == "High"]
    low_df = df[df["rating"] == "Low"]

    stats: dict[str, dict[str, float]] = {}
    for col in FEATURE_COLS:
        series = df[col]
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "high_mean": float(high_df[col].mean()),
            "low_mean": float(low_df[col].mean()),
            "high_median": float(high_df[col].median()),
            "low_median": float(low_df[col].median()),
        }
    return stats


def train_and_save(
    data_path: Path | str = DATA_PATH,
    artifact_path: Path | str = ARTIFACT_PATH,
) -> dict[str, Any]:
    """Train candidate models, evaluate on 80/20 stratified split, and persist artifact."""
    data_path = Path(data_path)
    artifact_path = Path(artifact_path)

    df = pd.read_csv(data_path)
    X = df[FEATURE_COLS]
    y = (df["rating"] == "High").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates: dict[str, Any] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    comparison: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        metrics = _evaluate_model(clf, X_test, y_test)
        trained_models[name] = clf
        comparison.append({"model": name, **metrics})

    selected_name = "Random Forest"
    selected_model = trained_models[selected_name]
    selected_metrics = next(
        row for row in comparison if row["model"] == selected_name
    )

    feature_importances = [
        {"feature": col, "label": FEATURE_META[col]["label"], "importance": float(imp)}
        for col, imp in zip(FEATURE_COLS, selected_model.feature_importances_)
    ]
    feature_importances.sort(key=lambda item: item["importance"], reverse=True)

    artifact: dict[str, Any] = {
        "model": selected_model,
        "model_name": selected_name,
        "sklearn_version": sklearn.__version__,
        "feature_cols": FEATURE_COLS,
        "metrics": selected_metrics,
        "model_comparison": comparison,
        "feature_importances": feature_importances,
        "feature_stats": _compute_feature_stats(df),
        "dataset_summary": {
            "total_rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "n_features": len(FEATURE_COLS),
            "class_counts": {str(k): int(v) for k, v in df["rating"].value_counts().items()},
        },
    }

    joblib.dump(artifact, artifact_path)
    return artifact


def load_or_train_artifact(
    artifact_path: Path | str = ARTIFACT_PATH,
    data_path: Path | str = DATA_PATH,
) -> dict[str, Any]:
    """Load saved model artifact if compatible; otherwise train and save it first."""
    artifact_path = Path(artifact_path)
    if artifact_path.exists():
        try:
            artifact = joblib.load(artifact_path)
            if artifact.get("sklearn_version") == sklearn.__version__:
                return artifact
        except Exception:
            pass
    return train_and_save(data_path=data_path, artifact_path=artifact_path)


def predict_student(artifact: dict[str, Any], inputs: dict[str, float | int]) -> dict[str, Any]:
    """Predict pass/fail outcome and probabilities for a single student profile."""
    row = {col: int(max(0, round(float(inputs.get(col, 0))))) for col in FEATURE_COLS}
    X_input = pd.DataFrame([row], columns=FEATURE_COLS)

    model = artifact["model"]
    probs = model.predict_proba(X_input)[0]
    prob_fail = float(probs[0])
    prob_pass = float(probs[1])
    pred_class = int(model.predict(X_input)[0])

    stats = artifact["feature_stats"]
    comparisons: list[dict[str, Any]] = []
    for col in FEATURE_COLS:
        val = row[col]
        high_med = stats[col]["high_median"]
        high_mean = stats[col]["high_mean"]
        low_med = stats[col]["low_median"]
        comparisons.append(
            {
                "feature": col,
                "label": FEATURE_META[col]["label"],
                "student_value": val,
                "high_median": high_med,
                "high_mean": round(high_mean, 2),
                "low_median": low_med,
                "delta_from_high_median": round(val - high_med, 2),
            }
        )

    return {
        "prediction": pred_class,
        "rating_label": "High" if pred_class == 1 else "Low",
        "outcome_text": (
            "LIKELY TO PASS (Rating: High)"
            if pred_class == 1
            else "LIKELY TO FAIL / AT-RISK (Rating: Low)"
        ),
        "prob_pass": prob_pass,
        "prob_fail": prob_fail,
        "inputs": row,
        "comparisons": comparisons,
    }


def format_student_comparison(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format student-vs-cohort feature comparisons into display-ready rows."""
    return [
        {
            "Predictor": row["label"],
            "Column": row["feature"],
            "Student Input": row["student_value"],
            "Passing (High) Median": row["high_median"],
            "Passing (High) Mean": row["high_mean"],
            "At-Risk (Low) Median": row["low_median"],
            "Diff vs. Passing Median": row["delta_from_high_median"],
        }
        for row in comparisons
    ]


def summarize_engagement_notes(comparisons: list[dict[str, Any]]) -> list[str]:
    """Return concise bullet-point observations comparing key activity counts to High medians."""
    key_features = {"Submission.Actions", "Create.Actions", "Read.Actions", "Days.Logged.In"}
    notes: list[str] = []
    for row in comparisons:
        if row["feature"] not in key_features:
            continue
        val = row["student_value"]
        high_med = row["high_median"]
        label = row["label"]
        if val < high_med:
            notes.append(
                f"{label} ({val}) is below the typical passing median ({high_med:.0f})."
            )
        elif val > high_med:
            notes.append(
                f"{label} ({val}) exceeds the typical passing median ({high_med:.0f})."
            )
    return notes


def format_model_comparison(comparison: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Format model benchmark metrics into display-ready rows."""
    return [
        {
            "Model": row["model"],
            "Accuracy": f"{row['accuracy']:.2%}",
            "ROC-AUC": f"{row['roc_auc']:.4f}",
            "Precision": f"{row['precision']:.2%}",
            "Recall": f"{row['recall']:.2%}",
            "F1-Score": f"{row['f1']:.2%}",
        }
        for row in comparison
    ]


def build_probability_chart(prob_pass: float, prob_fail: float) -> Any:
    """Create a horizontal probability bar chart for Pass vs Fail."""
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Bar(
                x=[prob_pass * 100, prob_fail * 100],
                y=["Likely to Pass (High)", "Likely to Fail / At-Risk (Low)"],
                orientation="h",
                marker_color=["#16a34a", "#dc2626"],
                text=[f"{prob_pass:.1%}", f"{prob_fail:.1%}"],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title="Prediction Confidence",
        xaxis=dict(title="Probability (%)", range=[0, 100]),
        yaxis=dict(autorange="reversed"),
        height=230,
        margin=dict(l=20, r=20, t=45, b=30),
    )
    return fig


THEME = {
    "navy": "#0b2545",
    "royal": "#13315c",
    "blue": "#1d4e89",
    "gold": "#c9a227",
    "pass": "#15803d",
    "fail": "#b91c1c",
    "muted": "#64748b",
    "grid": "#e5e9f2",
    "font": "Inter, Segoe UI, Roboto, sans-serif",
}


def _apply_chart_theme(fig: Any, height: int, title: str | None = None) -> Any:
    """Apply the shared academic colour palette and typography to a Plotly figure."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=THEME["font"], color="#1f2937", size=13),
        margin=dict(l=10, r=20, t=50 if title else 20, b=20),
    )
    if title:
        fig.update_layout(
            title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=THEME["navy"]))
        )
    return fig


def build_probability_gauge(prob_pass: float) -> Any:
    """Create a semicircular gauge showing the probability of passing."""
    import plotly.graph_objects as go

    bar_color = THEME["pass"] if prob_pass >= 0.5 else THEME["fail"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(prob_pass * 100, 1),
            number=dict(suffix="%", font=dict(size=40, color=THEME["navy"])),
            title=dict(text="Probability of Passing", font=dict(size=14, color=THEME["muted"])),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor=THEME["muted"], ticksuffix="%"),
                bar=dict(color=bar_color, thickness=0.28),
                bgcolor="white",
                borderwidth=0,
                steps=[
                    dict(range=[0, 50], color="#fdecec"),
                    dict(range=[50, 100], color="#e7f6ec"),
                ],
                threshold=dict(line=dict(color=THEME["gold"], width=4), thickness=0.85, value=50),
            ),
        )
    )
    _apply_chart_theme(fig, height=260)
    fig.update_layout(margin=dict(l=30, r=30, t=40, b=10))
    return fig


def build_feature_importance_chart(importances: list[dict[str, Any]]) -> Any:
    """Create a horizontal bar chart of Random Forest feature importances."""
    import plotly.express as px

    imp_df = pd.DataFrame(importances).sort_values("importance", ascending=True)
    fig = px.bar(
        imp_df,
        x="importance",
        y="label",
        orientation="h",
        text=imp_df["importance"].map(lambda v: f"{v:.1%}"),
        labels={"importance": "Relative Importance", "label": ""},
    )
    fig.update_traces(
        marker_color=THEME["royal"],
        marker_line_width=0,
        textposition="outside",
        textfont=dict(color=THEME["navy"]),
    )
    _apply_chart_theme(fig, height=380, title="Which predictors matter most?")
    fig.update_layout(
        xaxis=dict(
            range=[0, max(imp_df["importance"]) * 1.25],
            tickformat=".0%",
            gridcolor=THEME["grid"],
            zeroline=False,
        ),
        bargap=0.35,
    )
    return fig


def build_confusion_matrix_chart(cm: list[list[int]], test_rows: int) -> Any:
    """Create an annotated Plotly heatmap for the 2x2 confusion matrix."""
    import plotly.express as px

    labels = ["At-Risk (Low)", "Pass (High)"]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale=[[0, "#eef2f9"], [1, THEME["royal"]]],
        labels=dict(x="Predicted", y="Actual", color="Students"),
    )
    fig.update_traces(textfont=dict(size=20))
    _apply_chart_theme(fig, height=380, title=f"Confusion matrix on {test_rows} test students")
    fig.update_layout(coloraxis_showscale=False)
    return fig


if __name__ == "__main__":
    result = train_and_save()
    print(f"Saved artifact to: {ARTIFACT_PATH}")
    print(f"Selected model: {result['model_name']}")
    print("Model comparison:")
    for row in result["model_comparison"]:
        print(
            f"  {row['model']:<20} | Acc={row['accuracy']:.4f} | "
            f"Prec={row['precision']:.4f} | Rec={row['recall']:.4f} | "
            f"F1={row['f1']:.4f} | ROC-AUC={row['roc_auc']:.4f}"
        )
    print("\nTop Feature Importances:")
    for item in result["feature_importances"]:
        print(f"  {item['feature']:<20}: {item['importance']:.4f}")
