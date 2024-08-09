from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset("imdb")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2ForSequenceClassification.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='longest', max_length=512)

encoded_dataset = dataset.map(preprocess_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch", 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3, 
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer
)

trainer.train()
eval_result = trainer.evaluate()
print(eval_result)
