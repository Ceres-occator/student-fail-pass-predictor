# University Student Pass/Fail Prediction System

A machine learning web application that predicts whether a university student is likely to **Pass (`High` rating)** or **Fail / Become At-Risk (`Low` rating)** from 8 Learning Management System (LMS) engagement predictors.

## Dataset & Model Summary

- **Dataset**: `augmented_balanced_dataset_m.csv` (3,602 balanced records: 1,801 `High`, 1,801 `Low`; 0 missing values)
- **Predictors**:
  - `Submission.Actions`
  - `Quiz.Actions`
  - `Forum.Actions`
  - `Create.Actions`
  - `Read.Actions`
  - `Update.Actions`
  - `Delete.Actions`
  - `Days.Logged.In`
- **Production Model**: `RandomForestClassifier(n_estimators=200, random_state=42)` evaluated on an 80/20 stratified holdout split (`2,881` train / `721` test):
  - **Accuracy**: `88.21%`
  - **ROC-AUC**: `0.9522`
  - **Precision**: `93.10%`
  - **Recall**: `82.50%`
  - **F1-Score**: `87.48%`

## Project Structure

- `train_model.py` — Trains and benchmarks candidate classifiers, generates `model_artifact.joblib`, and provides shared inference/chart utilities.
- `app.py` — Faculty-facing **Streamlit** web application, "Student Success Check" (auto-trains if `model_artifact.joblib` is missing).
- `dash_app.py` — The same faculty-facing interface built with **Dash** (`server = app.server` exposed for Gunicorn); styles live in `assets/`.
- `faculty_content.py` — Plain-language labels, result wording, and follow-up suggestions shared by both apps, so teachers never see technical terms (e.g. `Read.Actions` → "Course materials opened", probability → "chance of passing").
- `verify_project.py` — Automated verification checks for model metrics, predictions, and app syntax/imports.
- `requirements.txt` & `Procfile` — Cloud deployment configuration.

## Running Locally

### Option 1: Streamlit (Recommended)
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Option 2: Dash
```bash
pip install -r requirements.txt
python train_model.py
python dash_app.py
```
Open `http://localhost:8050` in your browser.

## Deploying to Obtain a Public URL for Submission

Because this environment does not have authenticated GitHub/cloud CLI credentials (`gh` is not installed), deploy one of the options below using your browser account to get a permanent public URL:

### A. Streamlit Community Cloud (Fastest, Free)
1. Push this project folder to a GitHub repository.
2. Go to [https://share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository/branch, and set **Main file path** to `app.py`.
4. Click **Deploy!** — Streamlit Cloud installs `requirements.txt` and publishes a public `https://*.streamlit.app` URL to submit.

### B. Render / Railway (Using Dash + `Procfile` or Streamlit)
1. Push this project folder to GitHub.
2. On [Render](https://render.com) or [Railway](https://railway.app), create a new **Web Service** linked to your repository.
3. Build command: `pip install -r requirements.txt`
4. Start command:
   - For Dash: `gunicorn dash_app:server` (already configured in `Procfile`)
   - For Streamlit: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Copy the generated public `https://...` URL for submission.

### C. Hugging Face Spaces (Streamlit SDK)
1. Create a new Space at [https://huggingface.co/new-space](https://huggingface.co/new-space) with **Streamlit** as the Space SDK.
2. Upload `app.py`, `faculty_content.py`, `train_model.py`, `augmented_balanced_dataset_m.csv`, `requirements.txt`, and `.streamlit/config.toml` (the app theme).
3. Copy the public Space URL (`https://huggingface.co/spaces/<user>/<space>`) for submission.
