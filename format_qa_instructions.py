import json
import pandas as pd

with open("data/processed/qa_data.json", "r") as f:
    qa_data = json.load(f)

formatted_data = []

for item in qa_data:
    doc = item["doc"]
    question = item["q"]
    answer = item["a"]
    
    text = f"""
### Document:
{doc}

### Question:
{question}

### Answer:
{answer}
"""
    
    formatted_data.append(text)

formatted_df = pd.DataFrame({
    "text": formatted_data
})


formatted_df.to_csv(
    "data/processed/qa_instructions.csv",
    index=False
)


print("Q&A instruction formatting completed successfully.")