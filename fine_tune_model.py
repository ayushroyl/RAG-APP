import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import Dataset

# Manually create the dataset
data = [
    {"Query": "What does the policy booklet say about making a claim?", "Response": "The booklet provides a detailed process on how to make a claim, including necessary contact numbers (0345 878 6261 for claims, 0800 328 9150 for windscreen claims, and 0345 246 2408 for motor legal helpline). It also lists the information required to process a claim, such as personal details, policy number, car registration number, and a description of the loss or damage."},
    {"Query": "How much will you pay if my car is damaged?", "Response": "If your car is damaged and covered under the policy, the insurance will pay the cost of repairing or replacing the car up to its UK market value at the time of the claim."},
    {"Query": "Who is covered to drive other cars under this policy?", "Response": "The certificate of motor insurance will indicate who is covered to drive other cars. Only injury to third parties or damage to their property is covered, not damage to the car being driven."},
    {"Query": "Am I covered if I leave my car unlocked or the keys in the car?", "Response": "No, the policy does not cover theft or attempted theft if the car is left unlocked, with keys or key fobs in, on, or attached to the car, with the engine running, or with a window or roof open."},
    {"Query": "What is not included in the cover?", "Response": "The policy does not cover mechanical or electrical failure, wear and tear, damage to tyres caused by braking, punctures, cuts, bursts, or breakdowns. It also excludes cover if the car is being used by someone not insured on the policy, someone disqualified from driving, or for a purpose not allowed by the policy."},
    {"Query": "Does Churchill have approved repairers?", "Response": "Yes, Churchill customers have access to a national network of approved repairers who will handle all aspects of the repair."},
    {"Query": "What is DriveSure?", "Response": "DriveSure is a telematics insurance product that captures how, when, and where the car is driven, based on driver-monitoring technology. It provides feedback on driving style and bases the premium on the driving record."},
    {"Query": "What is the difference between commuting and business use?", "Response": "Business use covers driving in connection with a business or employment, while commuting covers driving to and from a permanent place of work, including to and from a car park, railway station, or bus stop as part of the journey."},
    {"Query": "Can I use my car abroad?", "Response": "The cover for using a car abroad depends on the type of policy and the destination. Full details can be found in the 'Where you can drive' section on page 31. A Green Card may be required for travel abroad."},
    {"Query": "Are electric car’s charging cables covered?", "Response": "Yes, home chargers and charging cables are considered accessories to the car and are covered under 'Section 2: Fire and theft' or 'Section 4: Accidental damage'. Coverage also includes accidents involving the charging cables when attached to the car, as long as due care is taken."},
    {"Query": "Is my electric car battery covered?", "Response": "The car's battery is covered if it is damaged as a result of an insured incident, regardless of whether the battery is owned or leased."},
    {"Query": "What is the definition of an accessory in the policy?", "Response": "Accessories are parts or products specifically designed to be fitted to the car, including electric car charging cables and home chargers. Some accessories may be considered modifications, so any changes to the car should be reported."},
    {"Query": "What does the term 'approved repairer' mean?", "Response": "An approved repairer is a repairer in the insurer's network of contracted repairers who is authorized to carry out repairs to the car following a claim under the policy."},
    {"Query": "What should you do if you receive a court notice related to a claim?", "Response": "Contact the insurer immediately if you receive any communication such as a court notice or threat of legal action. Provide any other relevant information, documents, or help needed to process the claim."},
    {"Query": "What is the procedure for repairing your car if it is damaged?", "Response": "If an approved repairer carries out the repairs, no estimate is needed, and the repair comes with a 5-year guarantee. If repairs are done by a chosen repairer with insurer approval, they are not guaranteed by the insurer."},
    {"Query": "How are windscreen repairs handled?", "Response": "The insurer may replace the car's glass with non-manufacturer glass of similar standard. If a non-approved supplier is used for windscreen repairs or replacement, insurer approval is not needed, but only a limited amount will be covered."},
    {"Query": "What happens if your car is written off?", "Response": "If the car is written off, the insurer will settle the claim and the policy's responsibilities will be met. No premium refund is provided if paid annually, and any balance will be paid to the legal owner if the car is leased or bought on hire purchase."},
    {"Query": "What does 'market value' mean in the context of the policy?", "Response": "Market value refers to the cost of replacing the car with another of the same make and model, and of a similar age, mileage, and condition at the time of the accident or loss."},
    {"Query": "What should be done to avoid increasing the claim amount?", "Response": "Do not do anything that would increase the claim amount without written permission from the insurer. This includes admitting liability or negotiating to settle any claim."},
    {"Query": "What is the 'policy' made up of?", "Response": "The policy consists of the policy booklet, car insurance details, certificate(s) of motor insurance, Green Flag breakdown cover booklet (if applicable), and DriveSure terms and conditions (if applicable)."},
    {"Query": "What are 'removable electronic equipment'?", "Response": "Removable electronic equipment refers to electronic devices designed to be fitted to and used in the car but can be removed when not in use. Speed assessment detection devices and personal portable electronics not specifically designed for car use are not covered."},
    {"Query": "How does the policy define 'vandalism'?", "Response": "Vandalism is defined as damage caused by a malicious and deliberate act."},
    {"Query": "What are the 'territorial limits' of the policy?", "Response": "The territorial limits include Great Britain, Northern Ireland, the Channel Islands, and the Isle of Man."},
    {"Query": "What is considered a 'track day'?", "Response": "A track day is when the car is driven on a racing track, airfield, or at an off-road event."},
    {"Query": "What does 'written off' mean?", "Response": "A car is considered written off if it is so badly damaged that it is no longer roadworthy or the cost to fix it would be uneconomical, based on its market value."},
    {"Query": "How does the policy define 'main driver'?", "Response": "The main driver is the person declared as the main user of the car and shown as the main driver on the car insurance details."},
    {"Query": "What is 'business use' in car insurance terms?", "Response": "Business use provides cover for driving in connection with a business or employment, as shown on the certificate of motor insurance."},
    {"Query": "What does 'Comprehensive with DriveSure' include?", "Response": "Comprehensive with DriveSure includes the same cover as a Comprehensive policy, with additional DriveSure terms and conditions."},
    {"Query": "Are tyres covered for damage?", "Response": "Damage to tyres caused by braking, punctures, cuts, or bursts is not covered under the policy."},
    {"Query": "What is covered under 'Section 1: Liability'?", "Response": "Section 1: Liability covers injury to third parties or damage caused to their property, but not to the car being driven."}
]

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
