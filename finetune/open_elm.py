import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, set_seed, get_constant_schedule
from trl import SFTTrainer, setup_chat_format, DataCollatorForCompletionOnlyLM
from storage.hugging_face_store import HuggingFaceModelStore
from datasets import load_dataset
import uuid, wandb
import torch

# Append directories to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'dataset')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'model')))

async def fine_tune_openELM(job_id, model_id, dataset_id):
  
  model = AutoModelForCausalLM.from_pretrained(
  model_id,
  trust_remote_code=True,
  device_map = "auto",
  torch_dtype = torch.float16,
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
    output_dir = f"out_{run_id}",
    evaluation_strategy = "steps",
    label_names = ["labels"],
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2,
    save_steps = 250,
    eval_steps = 250,
    logging_steps = 1,
    learning_rate = lr,
    num_train_epochs = 3,
    lr_scheduler_type = "constant",
    optim = 'paged_adamw_8bit',
    bf16 = False,
    gradient_checkpointing = True,
    group_by_length = True,
  )
  
  trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    eval_dataset = dataset['test'],
    data_collator = DataCollatorForCompletionOnlyLM(
        instruction_template = "<|im_start|>user",
        response_template = "<|im_start|>assistant",
        tokenizer = tokenizer,
        mlm = False),
    max_seq_length = 2048,
    dataset_kwargs = dict(add_special_tokens = False),
    args = training_arguments,
    )
  
  trainer.train()

  # Upload the model and tokenizer to Hugging Face Hub using the job ID for naming the repo
  store = HuggingFaceModelStore()
  await store.upload_model(model, tokenizer, job_id)