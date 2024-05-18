import os
import sys

# Append directories to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'finetune')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'finetune/runpod')))

import argparse
import asyncio
import json
import time

import transformers
from bert_fine_tune import fine_tune_bert
from gpt_fine_tune import fine_tune_gpt
from open_elm import fine_tune_openELM
from halo import Halo  # Ensure Halo is installed and imported
from helpers import fetch_and_save_job_details, fetch_jobs, update_job_status, submit_to_runpod
from llama_fine_tune import fine_tune_llama
from t5_fine_tune import fine_tune_t5
from pipeline import generate_pipeline_script

async def process_job(job_details, run_on_runpod=False, runpod_api_key=None):
    print("Transformer version " + transformers.__version__)
    model_id = job_details['baseModel']
    dataset_id = job_details['huggingFaceId']
    job_id = job_details['jobId']

    try:
        if run_on_runpod:
            if runpod_api_key is None:
                raise ValueError("RunPod API key is required when running on RunPod.")
            script_path = f"pipeline_script_{job_id}.py"
            generate_pipeline_script(job_details, script_path)
            submit_to_runpod(script_path, runpod_api_key)
        else:
            await fine_tune_openELM(job_id, model_id, dataset_id)

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
                await process_job(
                    job_details,
                    run_on_runpod=args.runpod,
                    runpod_api_key=args.runpod_api_key
                )
                spinner.succeed(f"Job {job_id} processed successfully")
        await asyncio.sleep(100)
        spinner.text = 'Waiting for jobs'
    spinner.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated training and uploading")
    parser.add_argument('--wallet_address', type=str, required=True)
    parser.add_argument('--runpod', action='store_true', help="Run the job on RunPod")
    parser.add_argument('--runpod_api_key', type=str, help="RunPod API key")
    args = parser.parse_args()
    asyncio.run(main(args))