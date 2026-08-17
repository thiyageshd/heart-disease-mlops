const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ShadingType,
  ImageRun, PageBreak, TableOfContents, PageNumber, Header, Footer,
} = require("docx");

const FIG = path.join(__dirname, "figures");
const img = (f) => fs.readFileSync(path.join(FIG, f));

const NAVY = "1a2a4f", BLUE = "2b6cb0", GREY = "555555";

// ---- helpers ----------------------------------------------------------
const H1 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 120 } });
const H2 = (t) => new Paragraph({ text: t, heading: HeadingLevel.HEADING_2, spacing: { before: 180, after: 80 } });
const P = (t, opts = {}) => new Paragraph({
  spacing: { after: 120, line: 276 },
  children: [new TextRun({ text: t, size: 22, ...opts })],
});
const bullet = (t) => new Paragraph({
  bullet: { level: 0 }, spacing: { after: 60 },
  children: [new TextRun({ text: t, size: 22 })],
});
const caption = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 160 },
  children: [new TextRun({ text: t, italics: true, size: 18, color: GREY })],
});
const figure = (file, w, h, cap) => [
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 },
    children: [new ImageRun({ type: "png", data: img(file), transformation: { width: w, height: h } })],
  }),
  caption(cap),
];

// shaded table helper
function table(headers, rows, widths) {
  const total = widths.reduce((a, b) => a + b, 0);
  const headRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: NAVY },
      margins: { top: 60, bottom: 60, left: 80, right: 80 },
      children: [new Paragraph({ children: [new TextRun({ text: h, bold: true, color: "FFFFFF", size: 20 })] })],
    })),
  });
  const bodyRows = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: ri % 2 ? "F2F5FA" : "FFFFFF" },
      margins: { top: 50, bottom: 50, left: 80, right: 80 },
      children: [new Paragraph({ children: [new TextRun({ text: String(c), size: 20, bold: i === 0 && ri >= 0 && r[0].includes && false })] })],
    })),
  }));
  return new Table({ columnWidths: widths, width: { size: total, type: WidthType.DXA }, rows: [headRow, ...bodyRows] });
}

const placeholder = (label) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 },
  border: { top: { style: BorderStyle.DASHED, size: 6, color: BLUE }, bottom: { style: BorderStyle.DASHED, size: 6, color: BLUE }, left: { style: BorderStyle.DASHED, size: 6, color: BLUE }, right: { style: BorderStyle.DASHED, size: 6, color: BLUE } },
  children: [new TextRun({ text: `[ SCREENSHOT: ${label} ]`, italics: true, color: BLUE, size: 20 })],
});

// ---- document ---------------------------------------------------------
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 2400 } }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Heart Disease Risk Prediction", bold: true, size: 52, color: NAVY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: "An End-to-End MLOps Pipeline", size: 32, color: BLUE })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 }, children: [new TextRun({ text: "MLOps Assignment 01 — AIMLCZG523", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: [new TextRun({ text: "Machine Learning Operations", size: 22, color: GREY })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Thiyagesh Dhandapani", size: 24, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "GitHub: https://github.com/<your-username>/heart-disease-mlops", size: 18, color: GREY })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// TOC
children.push(H1("Table of Contents"));
children.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 1. Overview
children.push(H1("1. Project Overview"));
children.push(P("This project builds and deploys a machine-learning system that predicts a patient's risk of heart disease from routine clinical measurements, and wraps that model in a complete, production-style MLOps lifecycle: reproducible data processing, tracked experimentation, automated testing, containerised serving, cloud-style deployment, and live monitoring."));
children.push(P("The emphasis is not only on model accuracy but on the engineering discipline around it — the parts that make a model dependable in production: reproducibility, automation, observability, and clean rollback."));
children.push(H2("Problem statement"));
children.push(P("Given 13 clinical features (age, sex, chest-pain type, resting blood pressure, cholesterol, and others), classify whether a patient has heart disease. The trained model is exposed as a JSON REST API and deployed to a Kubernetes cluster with request monitoring."));
children.push(H2("Dataset"));
children.push(P("Heart Disease UCI dataset (Cleveland processed subset): 303 records, 13 features, and a binary target (presence/absence of heart disease). After cleaning, 302 records remain; the classes are mildly imbalanced (~54% positive)."));

// 2. Architecture
children.push(H1("2. System Architecture"));
children.push(P("The pipeline spans three lanes — development, CI/CD, and serving/operations — each feeding the next. Training produces a single serialised pipeline artifact that is baked into a Docker image; CI validates and rebuilds on every push; and the container is deployed to Kubernetes and monitored with Prometheus and Grafana."));
children.push(...figure("architecture.png", 620, 349, "Figure 1. End-to-end MLOps architecture across development, CI/CD, and serving/ops."));

// 3. Data & EDA
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("3. Data Acquisition & Exploratory Data Analysis"));
children.push(P("Data is fetched by src/download_data.py, which targets the canonical UCI repository with a stable mirror fallback and writes data/heart_disease_raw.csv. Cleaning is deterministic and applied before any train/test split:"));
children.push(bullet("Invalid encoding artifacts (ca == 4, thal == 0) are coerced to missing for later imputation."));
children.push(bullet("Exact duplicate rows are dropped (one found)."));
children.push(bullet("Statistical imputation and scaling are deferred to the fitted pipeline so they are learned only from training folds — avoiding data leakage."));
children.push(H2("Class balance"));
children.push(...figure("class_balance.png", 340, 260, "Figure 2. Target class distribution — mildly imbalanced (~54% positive)."));
children.push(P("Because the imbalance is mild, accuracy is not badly misleading, but precision, recall and ROC-AUC are still reported. In a clinical screening context, recall (catching true cases) is prioritised."));
children.push(H2("Feature distributions and correlations"));
children.push(...figure("numeric_distributions.png", 560, 300, "Figure 3. Numeric feature distributions split by target."));
children.push(...figure("correlation_heatmap.png", 520, 400, "Figure 4. Feature correlation matrix. Strongest target associations: ca, exang, oldpeak, cp, thalach."));

// 4. Feature engineering & modelling
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("4. Feature Engineering & Model Development"));
children.push(P("Features are grouped and transformed inside a single scikit-learn ColumnTransformer, ensuring identical processing at train and inference time:"));
children.push(bullet("Numeric (age, trestbps, chol, thalach, oldpeak): median imputation then standard scaling."));
children.push(bullet("Categorical (cp, restecg, slope, thal, ca): most-frequent imputation then one-hot encoding."));
children.push(bullet("Binary (sex, fbs, exang): passed through unchanged (already 0/1)."));
children.push(P("Two classifiers were trained and tuned with 5-fold stratified GridSearchCV, scored on ROC-AUC: Logistic Regression (a strong, interpretable linear baseline) and Random Forest (captures non-linear interactions)."));
children.push(H2("Results"));
children.push(table(
  ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
  [
    ["Logistic Regression (selected)", "0.820", "0.824", "0.848", "0.836", "0.894"],
    ["Random Forest", "0.770", "0.744", "0.879", "0.806", "0.893"],
  ],
  [3000, 1300, 1300, 1200, 1000, 1200],
));
children.push(new Paragraph({ spacing: { after: 120 } }));
children.push(P("Logistic Regression was selected on ROC-AUC and accuracy; its tuning independently chose class_weight='balanced', directly addressing the imbalance. Random Forest achieved higher recall (0.879), a defensible alternative when minimising missed cases matters most."));
children.push(...figure("roc_logistic_regression.png", 330, 300, "Figure 5. ROC curve for the selected Logistic Regression model."));
children.push(...figure("cm_logistic_regression.png", 300, 275, "Figure 6. Confusion matrix (test set) for the selected model."));

// 5. Experiment tracking
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("5. Experiment Tracking (MLflow)"));
children.push(P("Every training run is logged to MLflow (SQLite backend) with hyper-parameters, cross-validated and test metrics, the ROC curve, the confusion matrix, a classification report, and the serialised model. Runs are compared in the MLflow UI:"));
children.push(P("mlflow ui --backend-store-uri sqlite:///mlflow.db", { font: "Consolas", size: 20 }));
children.push(placeholder("MLflow run comparison view (both runs, metrics columns)"));
children.push(placeholder("MLflow single-run page showing logged params, metrics, and artifacts"));

// 6. Packaging & reproducibility
children.push(H1("6. Model Packaging & Reproducibility"));
children.push(P("The winning pipeline (preprocessing + classifier in one object) is persisted to models/heart_pipeline.joblib, with a models/model_metadata.json sidecar recording the winner, metrics, best parameters, feature order, and cleaning notes. All dependencies are pinned in requirements.txt. Because the API imports this exact artifact, there is zero train/serve skew."));

// 7. Serving API
children.push(H1("7. Model Serving API (FastAPI)"));
children.push(P("The model is served by a FastAPI application exposing:"));
children.push(bullet("GET /health — liveness probe returning model metadata (used by Kubernetes)."));
children.push(bullet("POST /predict — single-patient prediction returning label, probability, and confidence."));
children.push(bullet("POST /predict/batch — multiple patients in one request."));
children.push(bullet("GET /metrics — Prometheus metrics."));
children.push(P("Input is validated with Pydantic (clinically sensible ranges), so malformed requests receive a clear 422 error. Interactive OpenAPI docs are available at /docs."));
children.push(placeholder("Swagger /docs page showing the endpoints"));
children.push(placeholder("Successful POST /predict response (JSON with prediction + confidence)"));

// 8. Containerisation
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("8. Containerisation (Docker)"));
children.push(P("The service is packaged in a slim python:3.11 image that installs serving-only dependencies, runs as a non-root user, and defines a container HEALTHCHECK against /health. Build and run:"));
children.push(P("docker build -t heart-disease-api .", { font: "Consolas", size: 20 }));
children.push(P("docker run -p 8000:8000 heart-disease-api", { font: "Consolas", size: 20 }));
children.push(placeholder("docker build output (successful image build)"));
children.push(placeholder("docker run + curl /predict returning a prediction"));

// 9. CI/CD
children.push(H1("9. CI/CD Pipeline (GitHub Actions)"));
children.push(P("On every push and pull request to main, the workflow runs: dependency install, ruff lint, dataset fetch, pytest (15 unit tests across preprocessing, the model artifact, and the API), a model-training smoke run, and artifact upload. The pipeline fails loudly on any lint or test error, satisfying the production-readiness requirement."));
children.push(placeholder("GitHub Actions run: all steps green (lint, test, train)"));
children.push(placeholder("pytest output showing 15 passed"));

// 10. Deployment
children.push(H1("10. Production Deployment (Kubernetes)"));
children.push(P("The container is deployed to a local Kubernetes cluster (Minikube) using manifests in k8s/: a Deployment (2 replicas, readiness/liveness probes on /health, resource requests/limits) and a LoadBalancer Service. Deploy with:"));
children.push(P("minikube image load heart-disease-api:latest", { font: "Consolas", size: 20 }));
children.push(P("kubectl apply -f k8s/", { font: "Consolas", size: 20 }));
children.push(P("minikube service heart-disease-api", { font: "Consolas", size: 20 }));
children.push(placeholder("kubectl get pods/svc showing running pods and the service"));
children.push(placeholder("Prediction against the deployed cluster endpoint"));

// 11. Monitoring
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("11. Monitoring & Logging"));
children.push(P("The API emits structured request logs and Prometheus metrics (request counts, latency histograms, response sizes per endpoint) at /metrics. A docker-compose stack brings up the API, Prometheus (scraping /metrics), and Grafana (auto-provisioned Prometheus datasource) together:"));
children.push(P("docker compose up --build", { font: "Consolas", size: 20 }));
children.push(placeholder("Prometheus targets page showing the API target UP"));
children.push(placeholder("Grafana dashboard with request-rate / latency panels"));

// 12. How to run
children.push(H1("12. How to Run (Quick Reference)"));
children.push(table(
  ["Step", "Command"],
  [
    ["Install", "pip install -r requirements.txt"],
    ["Get data", "python src/download_data.py"],
    ["EDA", "jupyter notebook notebooks/01_eda.ipynb"],
    ["Train", "python src/train.py"],
    ["Track", "mlflow ui --backend-store-uri sqlite:///mlflow.db"],
    ["Serve", "uvicorn src.api:app --port 8000"],
    ["Test", "pytest tests/"],
    ["Docker", "docker build -t heart-disease-api . && docker run -p 8000:8000 heart-disease-api"],
    ["Monitor", "docker compose up --build"],
    ["Deploy", "kubectl apply -f k8s/"],
  ],
  [1800, 7200],
));

// 13. Conclusion
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(H1("13. Conclusion"));
children.push(P("The delivered system demonstrates a complete MLOps lifecycle: from a reproducible data pipeline and tracked experimentation, through automated testing and containerised serving, to Kubernetes deployment and live monitoring. The selected Logistic Regression model achieves 0.89 ROC-AUC with balanced precision and recall, and the surrounding engineering ensures the model can be retrained, validated, deployed, and observed reliably."));
children.push(H2("Repository"));
children.push(P("https://github.com/<your-username>/heart-disease-mlops", { color: BLUE }));

// ---- assemble ---------------------------------------------------------
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 30, bold: true, color: NAVY }, paragraph: { spacing: { before: 240, after: 120 } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 25, bold: true, color: BLUE }, paragraph: { spacing: { before: 160, after: 80 } } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Heart Disease MLOps — Assignment 01", size: 16, color: "999999" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: "999999" })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "MLOps_Assignment01_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote", out, buf.length, "bytes");
});
