import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Manually create the dataset
data = {
    "query": [
        "What is covered under comprehensive insurance?",
        "How can I make a claim?",
        "Are there any exclusions to the policy?",
        "What is the windscreen cover limit?",
        "What should I do if my car is stolen?"
    ],
    "response": [
        "Comprehensive insurance covers damage to your car, theft, fire, and accidental damage. (Page 8)",
        "You can make a claim by contacting our claims team at 0345 603 3599, available 24/7. (Page 25)",
        "Yes, general exclusions include damage caused by war, terrorism, and nuclear risks. (Page 10)",
        "Windscreen cover is provided up to £100 with a £25 excess. (Page 14)",
        "Report the theft to the police and then contact our claims team as soon as possible. (Page 26)"
    ]
}

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

# Tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a padding token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['query'], truncation=True, padding='max_length', max_length=128)
    outputs = tokenizer(examples['response'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = outputs['input_ids']
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")
