
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
    job_details = {'baseModel': 'apple/OpenELM-450M', 'seed': 'Random', 'huggingFaceId': 'g-ronimo/oasst2_top4k_en', 'fineTuningType': 'text-generation', 'suffix': 'Apple', 'numberOfEpochs': '0', 'userId': 'dZhabDtALkOxnJYluQUrCBvRrGd2', 'batchSizeAuto': 'true', 'numberOfEpochsAuto': 'true', 'validationDataOption': 'none', 'trainingDataOption': 'selectExisting', 'modelParams': '', 'validationCriteria': '', 'learningRateMultiplier': '0', 'expectedOutcome': 'asdcfvgbhnjhbgvtfdcesxdcfvg', 'learningRateAuto': 'true', 'batchSize': '0', 'status': 'pending', 'createdAt': {'_seconds': 1715705455, '_nanoseconds': 36000000}, 'scriptPath': 'scripts/yLjl4DPXGR5QgDFxK781/2024-05-14T16-50-55.183Z-script.py', 'validationFilePath': '', 'trainingFilePath': '', 'scriptUrl': 'https://storage.googleapis.com/echo-fe663.appspot.com/scripts/yLjl4DPXGR5QgDFxK781/2024-05-14T16-50-55.183Z-script.py?GoogleAccessId=firebase-adminsdk-x8w53%40echo-fe663.iam.gserviceaccount.com&Expires=16447006800&Signature=gW1lxb6iEagFE1y4ISCLRTo558i1ZI7DZbjtdhju7y%2BDHCkqRH1vhWJ4Bolwm0F06PGZQSrgG63DNo%2Fo9yWq18C1zYyMrXVew6oYZqm9PjRntWT6Ss57UmyGsBLI7jeZqm6BgIxaO93pluI6WQKRCWZDAjU7BmOa1qbx%2FXl%2FTx0wDzMmvKm%2F58ZwgfSURJiXPgCR3OsZgzSU3qrqcFZ81I6%2Fd7N1T3mcriLypl1k6xOps0yd01q8xJ27E5dxc1qKDI%2FodvoN1RA8FRbtwPLjyatkWvsT%2BGnavfrLx6sCWYqYLgKZ%2BxTNQHYSLvUZlouWgHZotrAGlV5ec0wbga%2BKhA%3D%3D', 'trainingFileUrl': '', 'validationFileUrl': '', 'jobId': 'yLjl4DPXGR5QgDFxK781'}
    fine_tune_openELM(job_details["jobId"], job_details["baseModel"], job_details["huggingFaceId"])
