import os
import nltk
import numpy as np
import pandas as pd
from datasets import load_metric, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer

# Define the parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
max_input_length = 247
max_target_length = 25
batch_size = 8
model_checkpoint = "t5-small"
metric = load_metric("rouge")
wandb_run_name = 'SOTitle-t5'
output_dir = 'models/' + wandb_run_name
train_dataset_path = 'dataset/SOTitle/train.csv'
valid_dataset_path = 'dataset/SOTitle/valid.csv'
args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.001,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=15,
    predict_with_generate=True,
    fp16=True,
    run_name=wandb_run_name,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True
)


def preprocess_function(examples):
    inputs = [doc for doc in examples["src"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


if __name__ == '__main__':
    # Load the dataset
    train_dataset = pd.read_csv(train_dataset_path, low_memory=False)
    train_dataset.dropna(axis=0, how='any', inplace=True)
    train_dataset = Dataset.from_pandas(train_dataset)
    val_dataset = pd.read_csv(valid_dataset_path)
    val_dataset.dropna(axis=0, how='any', inplace=True)
    val_dataset = Dataset.from_pandas(val_dataset)

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(output_dir)
