# Project Memory — Heart Disease MLOps

Context file for AI coding assistants (Claude Code / Cursor / Copilot) working in this repo.
Read this first to understand what the project is, how it's structured, and the decisions already made.

---

## 1. What this project is

An end-to-end **MLOps pipeline** for **MLOps Assignment 01 (AIMLCZG523, BITS Pilani)**.
It trains a binary classifier that predicts **heart-disease risk** from 13 clinical features,
and wraps it in the full production lifecycle: reproducible preprocessing, tracked
experimentation, automated testing, containerised serving, Kubernetes deployment, and monitoring.

**This project does NOT call any LLM / OpenAI / Anthropic API.** It is plain scikit-learn +
FastAPI. It runs fully locally and free after `pip install -r requirements.txt`.

---

## 2. Dataset

- **Source:** authentic UCI Heart Disease dataset #45, Cleveland processed subset
  (`data/processed.cleveland.data`, committed to the repo).
- 303 records, 13 features, binary target.
- Original encodings are preserved: `cp∈{1,2,3,4}`, `thal∈{3,6,7}`, `ca∈{0,1,2,3}`,
  raw target `num∈{0-4}` (binarised to 0/1: 0 = no disease, >0 = disease).
- **6 genuine missing values** (marked `?` in source → NaN) in `ca` and `thal`, imputed
  inside the pipeline (NOT dropped).
- Target balance after binarising: **164 no-disease / 139 disease** (~54/46, near-balanced).

---

## 3. Repository structure

```
heart-disease-mlops/
├── data/
│   ├── processed.cleveland.data   # authentic UCI raw file (committed)
│   └── heart_disease_raw.csv      # generated 14-col CSV (download script output)
├── notebooks/
│   ├── 01_eda.ipynb               # EDA (executed, figures saved to report/figures)
│   └── 02_modeling.ipynb          # modelling narrative + results
├── src/
│   ├── __init__.py
│   ├── download_data.py           # fetch data (local file → UCI → mirror fallback)
│   ├── preprocess.py              # SHARED cleaning + sklearn ColumnTransformer
│   ├── train.py                   # train 2 models, MLflow, save best pipeline
│   └── api.py                     # FastAPI serving app
├── tests/
│   ├── test_preprocess.py         # data pipeline tests
│   └── test_api.py                # model artifact + API endpoint tests
├── models/
│   ├── heart_pipeline.joblib      # final fitted pipeline (preprocess + clf)
│   └── model_metadata.json        # winner, metrics, params, feature order
├── k8s/
│   ├── deployment.yaml            # 2 replicas, /health probes, resource limits
│   ├── service.yaml               # LoadBalancer
│   └── ingress.yaml               # optional host routing
├── monitoring/
│   ├── prometheus.yml             # scrapes api:8000/metrics
│   └── grafana/provisioning/…     # auto-provisioned Prometheus datasource
├── report/
│   ├── MLOps_Assignment01_Report.docx   # 12-page report (screenshot placeholders)
│   ├── build_report.js            # docx generator (docx-js)
│   └── figures/                   # all generated plots + architecture diagram
├── .github/workflows/ci.yml       # lint → test → train → upload artifact
├── Dockerfile                     # slim, non-root, HEALTHCHECK
├── docker-compose.yml             # API + Prometheus + Grafana stack
├── requirements.txt               # full dev/training deps (pinned)
├── requirements-serve.txt         # slim serving-only deps (for Docker)
├── pyproject.toml                 # ruff + pytest config
└── README.md
```

---

## 4. Key design decisions (do not undo these)

1. **Single shared preprocessing pipeline.** `src/preprocess.py` is imported by BOTH
   `train.py` and (via the unpickled joblib) `api.py`. This guarantees **zero train/serve
   skew** — identical imputation, scaling, and encoding at train and inference time.
   This is the most important architectural choice; keep it intact.

2. **Imputation lives inside the fitted pipeline**, not in standalone cleaning. Deterministic
   steps (coerce bad codes, drop dups) happen in `clean_data()` before the split; statistical
   steps (median/mode impute, scale) are learned only on training folds → no data leakage.

3. **Feature typing** (in `preprocess.py`):
   - numeric: `age, trestbps, chol, thalach, oldpeak` → median impute + standard scale
   - categorical: `cp, restecg, slope, thal, ca` → mode impute + one-hot (`handle_unknown="ignore"`)
   - binary: `sex, fbs, exang` → passthrough
   One-hot means the exact code values (e.g. `thal∈{3,6,7}`) don't matter.

4. **MLflow uses a SQLite backend** (`sqlite:///mlflow.db`), not the deprecated file store.
   Model logging uses `serialization_format="cloudpickle"` to avoid skops strict-typing errors.

5. **Model selection** by test ROC-AUC. Two candidates: Logistic Regression + Random Forest,
   each tuned with 5-fold **stratified** GridSearchCV.

6. **API validation** via Pydantic with clinically sensible ranges → malformed input returns 422.

---

## 5. Current results (authentic UCI data)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (selected)** | 0.885 | 0.839 | 0.929 | 0.881 | **0.968** |
| Random Forest | 0.885 | 0.839 | 0.929 | 0.881 | 0.950 |

Winner: **Logistic Regression** (higher ROC-AUC + simpler/interpretable). Recall 0.93 is
strong — prioritised because this is a screening use case.

---

## 6. Common commands

```bash
# setup
pip install -r requirements.txt

# data → train → track
python src/download_data.py
python src/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     # http://localhost:5000

# test + lint
pytest tests/                    # expect 15 passed
ruff check src tests

# serve
uvicorn src.api:app --port 8000  # docs at http://localhost:8000/docs

# example request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d @sample_request.json

# docker
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api

# full monitoring stack
docker compose up --build        # Prometheus :9090, Grafana :3000 (admin/admin)

# kubernetes (minikube running)
minikube image load heart-disease-api:latest
kubectl apply -f k8s/
minikube service heart-disease-api
```

---

## 7. API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + model metadata (K8s probes use this) |
| POST | `/predict` | single patient → prediction, label, probability, confidence |
| POST | `/predict/batch` | list of patients |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | Swagger UI |

`/predict` request body = 13 fields: `age, sex, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal` (see `sample_request.json`).

---

## 8. Conventions

- Python 3.11, ruff for lint (config in `pyproject.toml`; `E402/E501/E702` ignored).
- Pinned dependency versions — keep them pinned.
- Tests must pass before any commit; CI fails loudly on lint/test errors.
- Keep serving deps in `requirements-serve.txt` minimal (drives Docker image size).
- Commit style: conventional prefixes (`feat:`, `chore:`, `data:`).

---

## 9. Status & remaining work

**Done (code):** all 9 assignment tasks are implemented and tested.

**Remaining (human-only, cannot be automated):**
- Run everything locally and capture **screenshots** for the report placeholders
  (`[ SCREENSHOT: … ]` markers): MLflow UI, Swagger /docs, /predict response,
  docker build+run, GitHub Actions green run, `kubectl get pods,svc`, Grafana dashboard.
- Push to GitHub: `git remote add origin <url> && git push -u origin main`.
- Record a short (3–5 min) pipeline walkthrough video.
- Personalise EDA/report commentary in own words (academic-integrity requirement).

**Academic integrity note:** the assignment penalises identical/templated submissions.
Any AI assistance should help implement and explain, but feature choices, EDA commentary,
and architecture rationale should be personalised by the author.

---

## 10. Gotchas

- Don't move imputation out of the pipeline — it will cause data leakage and train/serve skew.
- The `ca==4 / thal==0` coercion in `preprocess.py` is a defensive no-op on authentic data
  (those artifacts only existed in an earlier pre-cleaned mirror). Safe to keep.
- If MLflow errors on model logging, confirm `serialization_format="cloudpickle"` is set.
- `mlflow.db`, `mlruns/`, `mlartifacts/` are gitignored — don't commit them.
- The model artifact `models/heart_pipeline.joblib` IS force-committed (grader needs it),
  even though `.gitignore` excludes `*.joblib` by default.
