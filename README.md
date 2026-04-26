# 📘 DocChat AI – Fine-Tuned LLM for Document Q&A & Toxic Comment Detection

> A locally runnable AI-powered system for document question answering and toxic comment detection built using **FastAPI**, **Streamlit**, **TinyLlama LoRA fine-tuning pipeline**, and managed using **uv**.

---

# 🚀 Project Overview

DocChat AI is an end-to-end AI application designed to:

- answer questions directly from document content
- detect toxic comments with confidence scores and explanations
- run completely locally using open-source tools
- avoid RAG and vector databases
- provide an interactive Streamlit-based UI
- support API-based inference using FastAPI

This project was developed as part of the **AI Developer Intern Assignment** by **Daivtech Solutions Pvt Ltd**.

---

# 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Base Model | TinyLlama |
| Fine-Tuning | LoRA using PEFT |
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Model Utilities | Hugging Face Transformers |
| Evaluation | scikit-learn, ROUGE-L |
| Package Manager | uv |
| Language | Python |

---

# 📂 Project Structure

```plaintext
docchat-ai/
│
├── api.py
├── ui.py
├── train.py
├── evaluation.py
├── README.md
├── pyproject.toml
├── requirements.txt
│
├── data/
│   └── processed/
│       ├── qa_instructions.csv
│       └── toxic_instructions.csv
│
├── models/
│   └── toxic-lora-adapter/
│
├── checkpoints/
│
├── reports/
│   ├── DocChat_AI_Project_Report.pdf
│   └── architecture-diagram.png
│
├── screenshots/
│   ├── swagger-api-docs.png
│   ├── streamlit-toxic-prediction.png
│   ├── streamlit-document-qa.png
│   └── evaluation-output.png
│
├── demo/
│   └── docchat-ai-demo.mp4
│
└── outputs/
```

---

# ✨ Features

## 📄 Document Question Answering

- paste plain-text documents
- ask questions from uploaded content
- returns answers based only on document content
- returns:

```text
Not mentioned in the document.
```

when answer is unavailable in the document

---

## ☣️ Toxic Comment Detection

Supports detection for:

- toxic
- obscene
- insult
- threat
- safe
- unknown

Returns:

- toxicity label
- confidence score
- explanation

### Example Response

```json
{
  "label": "insult",
  "confidence": 0.89,
  "explanation": "This comment contains rude and disrespectful language."
}
```

---

## ⚡ Edge Case Handling

The system handles:

- empty comments
- empty documents
- empty questions
- non-English comments
- long comments
- long documents
- API offline handling
- confidence threshold filtering
- batch prediction requests
- corrupted or empty dataset rows

---

# 🧠 Fine-Tuning Pipeline

## Dataset Preparation

### Q&A Dataset

Structured Q&A instruction formatting used:

```json
{
  "doc": "...",
  "q": "...",
  "a": "..."
}
```

---

### Toxic Comment Dataset

Toxic comments converted into instruction-style formatting.

Preprocessing included:

- text cleaning
- label formatting
- instruction generation
- empty row filtering

---

## LoRA Fine-Tuning

Implemented using:

- PEFT
- Hugging Face Transformers

Features implemented:

- CPU fallback support
- LoRA configuration
- dataset tokenization
- checkpoint saving
- adapter saving

---

# 🔌 API Endpoints

## POST `/predict`

Returns toxicity prediction.

### Request

```json
{
  "comment": "You are stupid"
}
```

### Response

```json
{
  "label": "insult",
  "confidence": 0.89,
  "explanation": "This comment contains rude and disrespectful language."
}
```

---

## POST `/ask`

Returns document-based answers.

### Request

```json
{
  "document": "Python is used for machine learning.",
  "question": "What is Python used for?"
}
```

### Response

```json
{
  "answer": "Python is used for machine learning."
}
```

---

## POST `/batch_predict`

Processes multiple comments together.

### Request

```json
{
  "comments": [
    "I hate you",
    "Hello friend"
  ]
}
```

---

# 🖥️ Streamlit UI

Features included:

- toxic comment prediction
- document Q&A interaction
- confidence threshold slider
- loading spinner
- persistent history using session state
- graceful API error handling

---

# ⚙️ Installation Guide

## Create Virtual Environment

```bash
uv venv
```

---

## Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
uv sync
```

---

# ▶️ Running the Project

## Start FastAPI Backend

```bash
uv run uvicorn api:app --reload
```

---

## Start Streamlit UI

```bash
uv run streamlit run ui.py
```

---

# 🧪 Evaluation Metrics

Implemented metrics:

| Metric | Purpose |
|---|---|
| F1 Score | Toxic classification evaluation |
| ROUGE-L | Q&A answer evaluation |

Run evaluation:

```bash
python evaluation.py
```

### Example Output

```text
F1 Score: 0.80
ROUGE-L Score generated successfully
```

---

# 🎥 Demo Video

Watch the complete project demonstration here:

```text
demo/docchat-ai-demo.mp4
```
---

# 📸 Screenshots

## Swagger API Documentation

![Swagger API Documentation]

---

## Streamlit Toxic Prediction

![Streamlit Toxic Prediction]

---

## Streamlit Document Q&A

![Streamlit Document Q&A]

---

## Evaluation Output

![Evaluation Output]

---

# 🔒 Constraints Followed

✅ Fully local execution

✅ No RAG implementation

✅ No vector database used

✅ Open-source tools only

✅ uv package manager used

---

# 📌 Design Decisions

- TinyLlama selected as lightweight open-source base model
- LoRA used for lightweight fine-tuning
- FastAPI used for modular backend APIs
- Streamlit used for rapid UI prototyping
- lightweight fallback inference logic used for rapid CPU-constrained API validation

---

# 📌 Future Improvements

- full LLM inference integration
- multi-document Q&A
- API authentication
- Hugging Face deployment
- BERTScore evaluation

---

# 📜 Assignment Reference

This project follows the requirements defined in the Daivtech Solutions AI Developer Intern assignment document.

---

# 👨‍💻 Author

**Mary Celine**

Aspiring AI-ML & Data Science Engineer

---

# ⭐ Final Note

This project demonstrates:

- dataset preprocessing
- LoRA fine-tuning pipeline design
- backend API development
- Streamlit frontend integration
- evaluation metric implementation
- edge-case handling
- local AI application architecture

Built with a focus on practical AI engineering workflows using open-source tools and local execution.
