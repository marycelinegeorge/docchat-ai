from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from rouge_score import rouge_scorer

# Case 1: Toxic Comment Classification Evaluation

print("Running Toxic Classification Evaluation ...")

true_labels =[
    "toxic",
    "safe",
    "insult",
    "safe",
    "threat"
]

predicted_labels = [
    "toxic",
    "safe",
    "insult",
    "toxic",
    "threat"
]


f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted"
)

print("\nF1 Score: ")
print(f1)

print("\nClassification Report")
print(
    classification_report(
        true_labels,
        predicted_labels
    )
)

# Case 1: ROUGE-L Evaluation

print("\nRunning ROUGE-L Evaluation...")

reference_answer = """

Python is used for machine learning and automation.
"""

generated_answer = """

Python is used for machine learning.
"""

scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

scores = scorer.score(
    reference_answer,
    generated_answer
)

print("\n ROUGE-L  Score: ")
print(scores["rougeL"])