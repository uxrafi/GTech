# Sequence Models for AI-Agent Execution Log Anomaly Detection:
## A Comparative Study Using the HDFS Benchmark

**CS7643: Deep Learning**  
**Georgia Institute of Technology**  
**Summer 2026**

### Team
- Umar Rafi
- Daniel Bird
- Sean McGee
- Ahmed Khamis

---

## Project Overview

This repository contains the implementation for our CS7643 Deep Learning project on sequence-based anomaly detection for AI-agent execution logs using the HDFS benchmark dataset.

We compare four supervised classification models:

- Logistic Regression (bag-of-events baseline)
- Recurrent Neural Network (LSTM)
- Transformer Encoder
- 1D Convolutional Neural Network (CNN)

Although public benchmarks for AI-agent execution logs remain limited, the HDFS benchmark provides ordered execution traces that serve as a reproducible proxy for many structural characteristics of autonomous agent workflows. The models are evaluated using Precision, Recall, F1-score, ROC-AUC, and PR-AUC to compare their effectiveness at detecting anomalous execution sequences.

The dataset comes from https://github.com/logpai/loghub

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.gatech.edu/urafi3/CS7643-Project.git
cd CS7643-Project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the HDFS Dataset

Download from: https://drive.google.com/file/d/1C1f6SxJq-n3hU3F7owUMRQL7DM6NtWSW/view?usp=drive_link
Place the unzipped data directory in `CS7643-Project`

---

## How to Run

### Logistic regression

```bash
python models/logistic_regression.py
```

Output metrics will be in `results/logreg`

### RNN with LSTM

```bash
python models/train_logistic_regression.py
```

Output metrics will be in `results/rnn`

### Transformer

```bash
python models/train_transformer.py
```

Output metrics will be in `results/transformer`

### CNN 1D

```bash
python models/train_cnn.py
```

Output metrics will be in `results/cnn`

## Paper

Final report written in LaTeX via Overleaf:  
https://www.overleaf.com/project/6a5cfa7130528a2235fdf7e9

