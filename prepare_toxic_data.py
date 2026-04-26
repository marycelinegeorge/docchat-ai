# Jicksaw Data Validation

import pandas as pd 

df = pd.read_csv("data/raw/train.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumns: ")
print(df.columns)

df = df.dropna(subset=["comment_text"])
df = df[df["comment_text"].str.strip() != ""]

print("\nAfter removing empty comments:")
print(df.shape)

labels = [
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "toxic"
]

def get_label(row):
    for label in labels:
        if row[label] == 1:
            return label
        
    return "safe"

df["label"] = df.apply(get_label, axis=1)

print("\nLabel Counts:")
print(df["label"].value_counts())


# EXPLAINATION for Bad Comments

explanations = {

    "safe": "This Comment is not harmful.",
    "obscene": " This Comment is offensive and not accepts the standard of morality and decency.",
    "toxic": "This Comment contaning harsh abusive language",
    "insult": "This Comment contains rude and disrespectful language.",
    "severe_toxic": "This Comment is highly abusive and has threatening language",
    "identity_hate": "This Comment has offensive language used against inherent identity factors.",
    "threat": "This Comment contain violence behaviour."
} 

def create_instructions(row):
    comment =row["comment_text"]
    label = row["label"]
    explanation =explanations[label]

    text = f"""### Comment:
{comment}

## OUTPUT
Label: {label}
Explanation: {explanation}
"""
    return text

df["instruction"]=df.apply(create_instructions, axis=1)
df["text"] = df["instruction"]

## Comment Verification
## print("\nSample Instruction:\n")
## print(df["instruction"].iloc[0])
## print("Toxic comment preprocessing completed successfully.")

processed_df = df[["text", "label"]]
processed_df.to_csv(
    "data/processed/toxic_instructions.csv",
    index=False
)


print("Toxic Instruction formatting completed successfully.")
