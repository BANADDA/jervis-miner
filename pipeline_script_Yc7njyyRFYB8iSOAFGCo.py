
import os
import sys
import subprocess

# Install required packages
required_packages = [
    "torch",
    "transformers",
    "datasets",
    "wandb",
    "trl",
    "huggingface_hub"
]

def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages(required_packages)

# Rest of the script
import uuid
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from datasets import load_dataset
from storage.hugging_face_store import HuggingFaceModelStore

def fine_tune_openELM(job_id, model_id, dataset_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "TinyPixel/Llama-2-7B-bf16-sharded",
        use_fast=False)

    set_seed(42)
    lr = 5e-5
    run_id = f"OpenELM-1_IB_LR-{lr}_OA_{str(uuid.uuid4())}"

    model, tokenizer = setup_chat_format(model, tokenizer)
    if tokenizer.pad_token in [None, tokenizer.eos_token]:
        tokenizer.pad_token = tokenizer.unk_token

    dataset = load_dataset(dataset_id)

    training_arguments = TrainingArguments(
        output_dir=f"out_{run_id}",
        evaluation_strategy="steps",
        label_names=["labels"],
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        save_steps=250,
        eval_steps=250,
        logging_steps=1,
        learning_rate=lr,
        num_train_epochs=3,
        lr_scheduler_type="constant",
        optim='paged_adamw_8bit',
        bf16=False,
        gradient_checkpointing=True,
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForCompletionOnlyLM(
            instruction_template="user",
            response_template="assistant",
            tokenizer=tokenizer,
            mlm=False),
        max_seq_length=2048,
        dataset_kwargs=dict(add_special_tokens=False),
        args=training_arguments,
    )

    trainer.train()

    store = HuggingFaceModelStore()
    store.upload_model(model, tokenizer, job_id)

if __name__ == "__main__":
    job_details = {'baseModel': 'apple/OpenELM-450M', 'seed': 'Random', 'huggingFaceId': 'g-ronimo/oasst2_top4k_en', 'fineTuningType': 'text-generation', 'suffix': 'Apple-OpenELM-450M', 'numberOfEpochs': '0', 'userId': 'dZhabDtALkOxnJYluQUrCBvRrGd2', 'batchSizeAuto': 'true', 'numberOfEpochsAuto': 'true', 'validationDataOption': 'none', 'trainingDataOption': 'selectExisting', 'modelParams': '', 'validationCriteria': '', 'learningRateMultiplier': '0', 'expectedOutcome': 'dcrtfvgbhnjkm njhbgfvtdcrxse', 'learningRateAuto': 'true', 'batchSize': '0', 'status': 'pending', 'createdAt': {'_seconds': 1715705296, '_nanoseconds': 451000000}, 'scriptPath': 'scripts/Yc7njyyRFYB8iSOAFGCo/2024-05-14T16-48-16.606Z-script.py', 'validationFilePath': '', 'trainingFilePath': '', 'scriptUrl': 'https://storage.googleapis.com/echo-fe663.appspot.com/scripts/Yc7njyyRFYB8iSOAFGCo/2024-05-14T16-48-16.606Z-script.py?GoogleAccessId=firebase-adminsdk-x8w53%40echo-fe663.iam.gserviceaccount.com&Expires=16447006800&Signature=pQKUXhkv5yUhzdhWDp3%2BE%2FdPWRzsopNu1WqnTidKvgxZpuXbKAR%2BvSsNSF8Kd47RbVqhUZBwmmOHujvdqIrjDMtLtEKoNy44SDCp%2FEAZsQ9tqSBO7t2PkcqRZ10ZRIhW1tLQS5ClfFVUDWFmJrii5U8aLRO%2F78dEyIIShTHlA4lP7RYky%2F%2BRpiYLwWjh%2B%2BrRoITLNIl445%2FVUTOWoXXK6nwDYaKoMARC6AJhN1UMGDpEMJCat4nFh8hifBCP7rRXdz9PIqIFq36VEIfa0Ih66HIfug%2BDR2dJvU2eniw8dtKCHDyM3C9Z%2BiMywWGu1zc1TJnMrO%2F9zET1%2Bj0j28Ht%2Bg%3D%3D', 'trainingFileUrl': '', 'validationFileUrl': '', 'jobId': 'Yc7njyyRFYB8iSOAFGCo'}
    fine_tune_openELM(job_details["jobId"], job_details["baseModel"], job_details["huggingFaceId"])
