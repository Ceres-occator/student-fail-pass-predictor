"""Verification suite for the Student Pass/Fail Prediction project."""

from __future__ import annotations

import importlib
import py_compile
from pathlib import Path

import sklearn

from train_model import (
    ARTIFACT_PATH,
    BASE_DIR,
    FEATURE_COLS,
    build_confusion_matrix_chart,
    build_feature_importance_chart,
    build_probability_chart,
    get_preset_profiles,
    load_or_train_artifact,
    predict_student,
)


def main() -> None:
    assert ARTIFACT_PATH.exists(), f"Missing artifact at {ARTIFACT_PATH}"
    artifact = load_or_train_artifact()

    assert artifact.get("sklearn_version") == sklearn.__version__
    assert artifact["feature_cols"] == FEATURE_COLS

    metrics = artifact["metrics"]
    assert metrics["accuracy"] >= 0.85, f"Accuracy too low: {metrics['accuracy']}"
    assert metrics["roc_auc"] >= 0.94, f"ROC-AUC too low: {metrics['roc_auc']}"
    print(
        f"[PASS] Model metrics verified: Accuracy={metrics['accuracy']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}"
    )

    presets = get_preset_profiles(artifact)
    high_res = predict_student(artifact, presets["passing"])
    assert high_res["prediction"] == 1, f"Expected High=1, got {high_res}"
    assert high_res["prob_pass"] >= 0.65, f"Unexpected low pass prob: {high_res}"

    low_res = predict_student(artifact, presets["failing"])
    assert low_res["prediction"] == 0, f"Expected Low=0, got {low_res}"
    assert low_res["prob_fail"] >= 0.65, f"Unexpected low fail prob: {low_res}"
    print(
        f"[PASS] Sample predictions verified: "
        f"High preset -> pass={high_res['prob_pass']:.3f}, "
        f"Low preset -> fail={low_res['prob_fail']:.3f}"
    )

    # Verify shared chart helpers construct valid Plotly figures
    fig_prob = build_probability_chart(high_res["prob_pass"], high_res["prob_fail"])
    fig_imp = build_feature_importance_chart(artifact["feature_importances"])
    fig_cm = build_confusion_matrix_chart(
        metrics["confusion_matrix"], artifact["dataset_summary"]["test_rows"]
    )
    assert len(fig_prob.data) == 1 and len(fig_imp.data) == 1 and len(fig_cm.data) == 1
    print("[PASS] Shared Plotly chart builders verified.")

    # Verify syntax compilation for all Python modules
    for filename in ("train_model.py", "faculty_content.py", "app.py", "dash_app.py"):
        file_path = Path(BASE_DIR) / filename
        py_compile.compile(str(file_path), doraise=True)
    print("[PASS] Syntax compilation passed for all app modules.")

    # Verify Dash app import and prediction callback
    dash_mod = importlib.import_module("dash_app")
    assert hasattr(dash_mod, "server")
    out = dash_mod.update_prediction(*[presets["passing"][c] for c in FEATURE_COLS], "Test", "Class")
    assert len(out) == 3 and "on-track" in out[0].className
    print("[PASS] dash_app imported cleanly and update_prediction callback executed.")

    # Verify Streamlit app import (and AppTest if streamlit is installed)
    app_mod = importlib.import_module("app")
    assert hasattr(app_mod, "main")
    try:
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(Path(BASE_DIR) / "app.py")).run(timeout=20)
        assert not at.exception, f"Streamlit AppTest raised exception: {at.exception}"
        print("[PASS] Streamlit AppTest rendered app.py without exceptions.")
    except ImportError:
        print("[WARN] streamlit not installed; skipped Streamlit AppTest.")


if __name__ == "__main__":
    main()
