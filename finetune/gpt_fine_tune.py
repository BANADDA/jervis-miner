import os
import sys

# Append directories to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'dataset')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'model')))

import transformers
from fetch_dataset import fetch_dataset
from load_model import load_model
from storage.hugging_face_store import HuggingFaceModelStore
from tokenize_data import tokenize_data
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, Trainer,
                          TrainingArguments)


async def fine_tune_gpt(job_id, model_id, dataset_id):
    """Fine-tunes GPT model for text generation and uploads it to Hugging Face Hub."""
    print(f"Transformer version: {transformers.__version__}")

    # Load tokenizer and model
    try:
        model, tokenizer = load_model(model_id, model_type="causal")
    except Exception as e:
        print(f"Failed to load model {model_id}: {str(e)}")
        raise

    # Fetch and tokenize the dataset
    try:
        dataset = fetch_dataset(dataset_id)
        tokenized_datasets = tokenize_data(tokenizer, dataset, model_type="gpt")
    except Exception as e:
        print(f"Failed to load dataset {dataset_id}: {str(e)}")
        raise

    # Set training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        save_strategy='epoch',
        evaluation_strategy='steps',
        eval_steps=500,
        logging_steps=500,
        load_best_model_at_end=True
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets.get('validation', None)  # Safely handle if no validation data is provided
    )

    # Start training
    trainer.train()

    # Upload the model and tokenizer to Hugging Face Hub using the job ID for naming the repo
    store = HuggingFaceModelStore()
    await store.upload_model(model, tokenizer, job_id)

# Example usage
# import asyncio

# model_id = "gpt2"
# dataset_id = "wikitext"
# config = "wikitext-2-raw-v1"  # Example dataset configuration
# asyncio.run(fine_tune_gpt("job1234", model_id, dataset_id, config))
