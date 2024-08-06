from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("imdb")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2ForSequenceClassification.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

print(f"Padding token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='longest', max_length=512)

encoded_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer
)

eval_result = trainer.evaluate()
print(f"Initial evaluation results: {eval_result}")
