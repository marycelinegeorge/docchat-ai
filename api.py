from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import re

# Future Model Integration
# LoRA fine-tuned model loading can be added here

# from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM

# model_path = "models/toxic-lora-adapter"

app = FastAPI(
    title = "DocChat AI Backend",
    description = "Document Q&A & Toxic Comment Detection"
)


# Request

class ToxicRequest(BaseModel):
    comment: str

class BatchToxicRequest(BaseModel):
    comments: List[str]

class QARequest(BaseModel):
    document: str
    question: str
    

# Toxic Label Explaination

toxic_explanation = {
    "safe": "This Comment is not harmful.",
    "obscene": " This Comment is offensive and not accepts the standard of morality and decency.",
    "toxic": "This Comment contaning harsh abusive language",
    "insult": "This Comment contains rude and disrespectful language.",
    "severe_toxic": "This Comment is highly abusive and has threatening language",
    "identity_hate": "This Comment has offensive language used against inherent identity factors.",
    "threat": "This Comment contain violence behaviour.",
    "unknown": "Unable to detect language or analuse the comment."
}


def is_english(text):
    english_pattern = r"^[a-zA-Z0-9\s.,!?'-]+$"
    return bool(re.match(english_pattern, text))


def detect_toxicity(comment):
    # Temporary lightweight inference logic used currently in this project
    # Used for fast API validation during CPU-constrained development

    # SCENARIO: Empty Comments

    if not comment.strip():

        return{
            "label": "safe",
            "confidence" : 1.0,
            "explanation": "No text to analyze"
        }

    # Scenario : Other Language Comments

    if not is_english(comment):

        return{
            "label": "unknown",
            "confidence" : 0.0,
            "explanation": "Non-English Comment to Detected."
        }

    # Scenario: Long Comment

    words = comment.split()

    if len(words) > 500:

        comment = " ".join(words[:500])
    

    # Scenario : Toxic Detection

    comment_lower = comment.lower()

    toxic_keyword = {

        "threat": ["kill", "destroy", "attack"],

        "insult" : ["idiot", "stupid", "loser"],

        "obscene": ["damn", "trash"],

        "toxic" : ["hate", "shutup"]
    }

    for label, words in toxic_keyword.items():

        for word in words:

            if word in comment_lower:

                return{
                    "label" : label,
                    "confidence" : 0.89,
                    "explanation" : toxic_explanation[label]
                }

    return{
        "label": "safe",
        "confidence": 0.95,
        "explanation": toxic_explanation["safe"]
    }


# Scenario: Empty Document Error

def answer_question(document, question):
    # Placeholder until full LLM inference integration

    if not document.strip():

        raise HTTPException(
            status_code = 400,
            detail = "Document is Empty."
        )
        

    # Scenario: Empty Question Error

    if not question.strip():

        raise HTTPException(
            status_code = 400,
            detail = "Question is empty."
        )


    # Scenario: Long Document uploading

    words = document.split()

    if len(words) > 2000:

        document = " ".join(words[:2000])


    document_lower = document.lower()

    question_words = question.lower().split()

    matched_words = 0


    for word in question_words:

        if word in document_lower:

            matched_words += 1


    # Common match handling

    if matched_words >= 2:

        return{
            "answer": document
        }


    return{
        "answer": "Not mentioned in the document."
    }


# CASE 1 : Health Check Endpoints

@app.get("/")

def health_check():

    return{
        "status": "API Running Successfully"
    }


# CASE 2: Toxic Prediction Endpoint

@app.post("/predict")

def predict_comment(request: ToxicRequest):

    result = detect_toxicity(request.comment)

    return result


# CASE 3: Batch Toxic Prediction Endpoint

@app.post("/batch_predict")

def batch_predict(request: BatchToxicRequest):

    results = []

    for comment in request.comments:

        prediction = detect_toxicity(comment)

        results.append(prediction)

    return{
        "results": results
    }


#CASE 4: Document Q&A Endpoints

@app.post("/ask")

def ask_question(request: QARequest):

    result = answer_question(
        request.document,
        request.question
    )

    return result
