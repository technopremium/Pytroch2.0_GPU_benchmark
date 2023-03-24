import os
from random import randrange
import numpy as np
from huggingface_hub import login, HfFolder
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import evaluate


def setup():
    login(
        token="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",  # ADD YOUR TOKEN HERE
        add_to_git_credential=True,
    )


def setup_trainer():
    # Load dataset
    dataset_id = "banking77"
    raw_dataset = load_dataset(dataset_id)

    # Load tokenizer
    model_id = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, return_tensors="pt")

    # Tokenize dataset
    raw_dataset = raw_dataset.rename_column("label", "labels")
    tokenized_dataset = raw_dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Load model
    labels = tokenized_dataset["train"].features["labels"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )

    # Load metric
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels, average="weighted")

    # Setup Trainer
    repository_id = "bert-base-banking77-pt2"
    training_args = TrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=3,
        bf16=True,
        torch_compile=True,
        optim="adamw_torch_fused",
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=200,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )

    return trainer


if __name__ == "__main__":
    setup()
    trainer = setup_trainer()
    trainer.train()
