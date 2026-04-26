import torch
# Libraries require for LoRA based LLM Fine-Tuning
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from peft import get_peft_model

from transformers import Trainer
from transformers import TrainingArguments

if torch.cuda.is_available():
    device = "cuda" 
else:
    device = "cpu"

print(f"Using Device: {device}")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading Model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

model.to(device)
## print("Model and Tokenizer Loaded Sucessfully!!!")

from datasets import load_dataset
print("Loading Training Datasets...")

qa_dataset = load_dataset("csv", data_files = "data/processed/qa_instructions.csv")
toxic_dataset = load_dataset("csv", data_files = "data/processed/toxic_instructions.csv")

print(qa_dataset)
print(toxic_dataset)

def tokenize_function(sample):

    tokens = tokenizer(
        sample["text"],
        truncation=True,
        padding="max_length",
        max_length=128 # reduced token length 512 
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens


# print("Dataset Tokenizing: ")
tokenized_qa = qa_dataset.map(tokenize_function)
# tokenized_qa = tokenized_qa.remove_columns(["text"])

tokenized_toxic = toxic_dataset.map(tokenize_function)
# tokenized_toxic = tokenized_toxic.remove_columns(["text", "label"])


print("Completed Tokenization !")

# Verifying

# sample = qa_dataset["train"][0]
# print(sample) # first draw 

# print("\n Text Value: ")
# text_value = sample["text"]
# print(text_value)

# print("\n Tokenized QA Sample: ")
# tokenized_sample = tokenized_qa["train"][0]
# print(tokenized_sample)

# print("\n Input IDs: ")
# tokens = tokenized_sample["input_ids"]
# print(len(tokens))

# print("\n Decoded Text: ")
# decoded_text = tokenizer.decode(tokens, skip_special_tokens=True)
# print(decoded_text)

small_qa = tokenized_qa["train"].select(range(9))
small_toxic = tokenized_toxic["train"].select(range(10)) # reduced training samples from 200 to 20

print("small datasets create successfully. ")

tokenized_qa = small_qa.map(tokenize_function)
tokenized_toxic = small_toxic.map(tokenize_function)

# tokenized_qa = tokenized_qa

print("LoRA Configuration...")

lora_config = LoraConfig(
    r=8, 
    lora_alpha = 16,
    target_modules = ["q_proj", "v_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print("LoRA Applied Successfully ...")

training_args = TrainingArguments(
    output_dir = "checkpoints",
    per_device_train_batch_size = 1,
    num_train_epochs = 1,
    max_steps = 10, # adding max_step for limiting the training steps
    logging_steps = 1, # reduced 5 to 1
    save_steps = 10,
    learning_rate = 2e-4,
    fp16 = False
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = small_toxic
)

print("Start Model Training...")

# trainer.train()

print("Training skipped due to CPU Hardware limitation.")

model.save_pretrained(
    "models/toxic-lora-adapter"
)

print("Model Adapter Saved Successfully...")