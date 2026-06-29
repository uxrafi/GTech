# SBKU — Anomaly Detection in Agent Execution Logs

**CS7643 Deep Learning | Georgia Tech | Summer 2026**  
**Team:** Umar Rafi, Daniel Bird, Sean McGee, Ahmed Khamis

---

## Project Overview

This project builds a deep learning system to detect anomalies in AI agent 
execution logs. The system analyzes sequences of log events from the HDFS 
dataset to identify unusual patterns that may indicate misbehavior, compliance 
violations, or system failures.

Normal agent execution follows predictable patterns. Our model learns what 
"normal" looks like, then flags deviations from expected behavior.

**ML Task:** Supervised binary sequence classification (Normal vs Anomalous)  
**Model:** LSTM-based recurrent neural network  
**Dataset:** HDFS (Hadoop Distributed File System) log dataset  

---

## Repository Structure

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/[your-repo]/sbku-anomaly-detection.git
cd sbku-anomaly-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the HDFS dataset

Download from: https://github.com/logpai/loghub  
Place raw log files in `data/raw/`

---

## How to Run

### Step 1 — Preprocess the data

```bash
python preprocessing/log_preprocessing.py
```

Outputs `.npy` sequence files to `data/sequences/`.

### Step 2 — Train the model

```bash
python models/train_rnn.py
```

Saves best checkpoint to `checkpoints/best_model.pt`.

### Step 3 — Evaluate the model

```bash
python models/evaluate_model.py
```

Outputs accuracy, precision, recall, F1, and AUC-ROC to `results/metrics/`.

### Step 4 — Run the API server

```bash
uvicorn api.api_server:app --reload
```

API available at `http://localhost:8000`

---

## API Endpoints

| Method | Endpoint | Owner | Description |
|--------|----------|-------|-------------|
| POST | `/logs/ingest` | Umar | Ingest raw log sequences |
| POST | `/analyze` | Umar | Run anomaly detection |
| GET | `/flags` | Sean | Retrieve flagged violations |
| GET | `/audit` | Ahmed | Retrieve audit log |
| GET | `/metrics` | Umar | Model performance metrics |

---

## Branching Convention

| Branch | Owner | Purpose |
|--------|-------|---------|
| `main` | All | Clean working code only |
| `daniel/preprocessing` | Daniel | Week 1 data pipeline |
| `umar/model` | Umar | Week 2 LSTM training |
| `sean/services` | Sean | Week 3 services |
| `ahmed/services` | Ahmed | Week 3 services |

Merge to `main` via pull request at the end of each week.  
Always pull from `main` before starting new work.

---

## Team Contributions

| Member | Role | Responsibilities |
|--------|------|-----------------|
| Umar Rafi | Model + API | LSTM architecture, training, evaluation, FastAPI server |
| Daniel Bird | Data + Testing | Preprocessing pipeline, integration testing, figures |
| Sean McGee | Services + API | LogIngestion, AnomalyDetection services, /flags endpoint |
| Ahmed Khamis | Services + API | Flagging, Audit services, /audit endpoint |

---

## Target Results

| Metric | Target |
|--------|--------|
| AUC-ROC | 95%+ |
| Precision | 90%+ |
| Recall | 90%+ |
| API Latency | < 200ms |

---

## Paper

Final report written in LaTeX via Overleaf:  
https://www.overleaf.com/read/fdjpfsdhztfp

Format: 4-6 page conference-style paper  
Submission: Gradescope (PDF)

