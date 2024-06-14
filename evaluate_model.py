from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the fine-tuned model
model = GPT2LMHeadModel.from_pretrained("fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_model")
qa_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Load the PDF text
with open("RAG_App\policy_text.txt", "r", encoding="utf-8") as f:
    pdf_text = f.read()

# Test data
test_data = {
    "query": [
        "What is covered under comprehensive insurance?",
        "How can I make a claim?",
        "Are there any exclusions to the policy?",
        "What is the windscreen cover limit?",
        "What should I do if my car is stolen?"
    ],
    "true_response": [
        "Comprehensive insurance covers damage to your car, theft, fire, and accidental damage. (Page 8)",
        "You can make a claim by contacting our claims team at 0345 603 3599, available 24/7. (Page 25)",
        "Yes, general exclusions include damage caused by war, terrorism, and nuclear risks. (Page 10)",
        "Windscreen cover is provided up to £100 with a £25 excess. (Page 14)",
        "Report the theft to the police and then contact our claims team as soon as possible. (Page 26)"
    ]
}

# Generate responses
predicted_responses = [qa_pipeline(q, max_length=50, num_return_sequences=1)[0]['generated_text'] for q in test_data['query']]

# Evaluation metrics
accuracy = accuracy_score(test_data['true_response'], predicted_responses)
precision = precision_score(test_data['true_response'], predicted_responses, average='micro')
recall = recall_score(test_data['true_response'], predicted_responses, average='micro')
f1 = f1_score(test_data['true_response'], predicted_responses, average='micro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
