import os
import sys

# Append directories to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'finetune')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))

# if not hasattr(sys, 'real_prefix') and (not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix):
#     print("This script is not running in a virtual environment. Please run it through the provided batch script.")
#     sys.exit(1)

import argparse
import asyncio
import json
import time

import transformers
from bert_fine_tune import fine_tune_bert
from gpt_fine_tune import fine_tune_gpt
from halo import Halo  # Ensure Halo is installed and imported
from helpers import fetch_and_save_job_details, fetch_jobs, update_job_status
from llama_fine_tune import fine_tune_llama
from t5_fine_tune import fine_tune_t5


async def process_job(job_details):
    print("Transfomer version" + transformers.__version__)
    """Processes a single job by dispatching it to the appropriate fine-tuning function based on the model type."""
    model_id = job_details['baseModel']
    dataset_id = job_details['huggingFaceId']
    job_id = job_details['jobId']

    try:
        # if 'gpt' in model_type.lower():
        await fine_tune_gpt(job_id, model_id, dataset_id)
        # elif 'bert' in model_type.lower():
        #     await fine_tune_bert(model_type, dataset_id, job_id)
        # elif 'llama' in model_type.lower():
        #     await fine_tune_llama(model_type, dataset_id, job_id)
        # elif 't5' in model_type.lower():
        #     await fine_tune_t5(model_type, dataset_id, job_id)
        # else:
        #     raise ValueError(f"No fine-tuning script available for model type: {model_type}")

        await update_job_status(job_id, 'completed')
    except Exception as e:
        print(f"Failed to process job {job_id}: {str(e)}")
        await update_job_status(job_id, 'failed')

async def main(args):
    spinner = Halo(text='Waiting for jobs', spinner='dots')
    spinner.start()
    while True:
        jobs = await fetch_jobs()
        if jobs:
            job_id = jobs[0]['id']
            spinner.text = f"Executing JobId: {job_id}"
            job_details_path = await fetch_and_save_job_details(job_id)
            if job_details_path:
                spinner.succeed(f"Job {job_id} fetched successfully")
                spinner.start(f"Processing JobId: {job_id}")
                with open(job_details_path, 'r') as file:
                    job_details = json.load(file)
                await process_job(job_details)
                spinner.succeed(f"Job {job_id} processed successfully")
        await asyncio.sleep(100)
        spinner.text = 'Waiting for jobs'
    spinner.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated training and uploading")
    parser.add_argument('--wallet_address', type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main(args))
