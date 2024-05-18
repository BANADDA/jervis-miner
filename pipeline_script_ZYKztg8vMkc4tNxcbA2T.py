
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
    job_details = {'baseModel': 'apple/OpenELM-450M', 'seed': 'Random', 'huggingFaceId': 'g-ronimo/oasst2_top4k_en', 'fineTuningType': 'text-generation', 'suffix': 'Apple_5', 'numberOfEpochs': '0', 'userId': 'dZhabDtALkOxnJYluQUrCBvRrGd2', 'batchSizeAuto': 'true', 'numberOfEpochsAuto': 'true', 'validationDataOption': 'none', 'trainingDataOption': 'selectExisting', 'modelParams': '', 'validationCriteria': '', 'learningRateMultiplier': '0', 'expectedOutcome': 'RTSJNHBVDHSBNFJBDSNFJBDSJF', 'learningRateAuto': 'true', 'batchSize': '0', 'status': 'pending', 'createdAt': {'_seconds': 1715707561, '_nanoseconds': 772000000}, 'scriptPath': 'scripts/ZYKztg8vMkc4tNxcbA2T/2024-05-14T17-26-01.925Z-script.py', 'validationFilePath': '', 'trainingFilePath': '', 'scriptUrl': 'https://storage.googleapis.com/echo-fe663.appspot.com/scripts/ZYKztg8vMkc4tNxcbA2T/2024-05-14T17-26-01.925Z-script.py?GoogleAccessId=firebase-adminsdk-x8w53%40echo-fe663.iam.gserviceaccount.com&Expires=16447006800&Signature=p6s%2BFS8xJYiA%2FHpFqrE7U5Nec8T71jE%2BgSoXLYO31dbDypeWzNjQR9zgsipe7MkDv%2BggKEtLsXiDldMijgHeQIVdh4CcGJk9YSnoBugl6KCWSa%2BKlX%2BbU%2BRtads2dna2tDOxMJfSryxduZCHSaimN%2FHRpkNHmqRYzLDdPK%2B%2Bb3U3wV%2BBswswXsIEdTff8GxE%2F8rIAOhJzyLnnXy%2FAzCUj9MGr1HqeJTdyYBe9jev1pDUnp4Ky6FSERL2ErRyu%2FhyB%2B3PGRKlSJqGt1IcwVkL%2FArAK1Fh15D6wuClWgi82fDs%2BqmbtpV82iqf93o0MxI17eFpZ9eIYmO5f23Ks%2BYD6w%3D%3D', 'trainingFileUrl': '', 'validationFileUrl': '', 'jobId': 'ZYKztg8vMkc4tNxcbA2T'}
    fine_tune_openELM(job_details["jobId"], job_details["baseModel"], job_details["huggingFaceId"])
