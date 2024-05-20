import os
import sys
import json
import time
import asyncio
import argparse
import aiohttp
import pyfiglet
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.table import Table
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
import threading
import keyboard
from dotenv import load_dotenv

# Append directories to sys.path for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'finetune')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './', 'auth')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'finetune/runpod')))

from bert_fine_tune import fine_tune_bert
from gpt_fine_tune import fine_tune_gpt
from open_elm import fine_tune_openELM
from helpers import fetch_and_save_job_details, fetch_jobs, update_job_status, submit_to_runpod
from llama_fine_tune import fine_tune_llama
from t5_fine_tune import fine_tune_t5
from pipeline import generate_pipeline_script
from auth import authenticate

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
TOKEN = os.getenv("TOKEN")
MINER_ID = os.getenv("MINER_ID")

console = Console()

pause_fetching = threading.Event()
stop_fetching = threading.Event()

def display_welcome_message():
    fig = pyfiglet.Figlet(font='slant')
    welcome_text = fig.renderText('Jarvis Miner')
    console.print(welcome_text, style="bold blue")
    console.print(Panel(Text("Welcome to Jarvis Miner System!", justify="center"), style="bold green"))

async def fetch_jobs():
    console.log("Waiting for training jobs")
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {TOKEN}'}
        async with session.get(f"{BASE_URL}/pending-jobs", headers=headers) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 401:
                return "Unauthorized"
            else:
                console.log(f"Failed to fetch jobs: {await response.text()}")
                return []

async def fetch_and_save_job_details(job_id):
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {TOKEN}'}
        async with session.post(f"{BASE_URL}/start-training/{job_id}", headers=headers,
                                json={'minerId': MINER_ID}) as response:
            if response.status == 200:
                job_details = await response.json()
                job_dir = os.path.join(os.getcwd(), 'jobs', job_id)
                os.makedirs(job_dir, exist_ok=True)
                details_path = os.path.join(job_dir, 'details.json')
                job_details['jobId'] = job_id
                with open(details_path, 'w') as f:
                    json.dump(job_details, f, indent=2)
                return details_path
            else:
                console.log(f"Failed to start training for job {job_id}: {await response.text()}")
                return None

async def update_job_status(job_id, status):
    url = f"{BASE_URL}/update-status/{job_id}"
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {TOKEN}', 'Content-Type': 'application/json'}
        async with session.patch(url, json={'status': status}, headers=headers) as response:
            try:
                if response.status == 200:
                    console.log(f"Status updated to {status} for job {job_id}")
                else:
                    response.raise_for_status()
            except aiohttp.ClientResponseError as err:
                console.log(f"Failed to update status for job {job_id}: {err}")
            except Exception as e:
                console.log(f"An error occurred: {e}")

async def process_job(job_details, run_on_runpod=False, runpod_api_key=None):
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
        console.log(f"Failed to process job {job_id}: {str(e)}")
        await update_job_status(job_id, 'failed')

async def handle_pause():
    global stop_fetching
    options = [
        Choice(name="Continue", value="continue"),
        Choice(name="Stop", value="stop")
    ]
    choice = await inquirer.select(
        message="Job fetching paused. Choose an option:",
        choices=options,
        default="continue",
    ).execute_async()

    if choice == "stop":
        stop_fetching.set()
    pause_fetching.clear()

def listen_for_key_presses():
    while True:
        if keyboard.is_pressed("p"):
            pause_fetching.set()
        if keyboard.is_pressed("c"):
            stop_fetching.set()

def run_keyboard_listener():
    listener_thread = threading.Thread(target=listen_for_key_presses, daemon=True)
    listener_thread.start()

async def main(args):
    display_welcome_message()

    username = input("Enter your username: ")
    password = input("Enter your password: ")

    token, miner_id = authenticate(username, password)
    if not token or not miner_id:
        console.log("[bold red]Authentication failed. Exiting...[/bold red]")
        sys.exit(1)

    global TOKEN, MINER_ID
    TOKEN, MINER_ID = token, miner_id

    progress_table = Table.grid(expand=True)
    progress_table.add_column(justify="center", ratio=1)
    progress_table.add_row(Panel("Executing...", title="Status", border_style="green"))

    run_keyboard_listener()

    with Live(progress_table, refresh_per_second=10, console=console) as live:
        while True:
            if stop_fetching.is_set():
                console.log("[bold red]Job fetching stopped by user.[/bold red]")
                break

            if pause_fetching.is_set():
                await handle_pause()

            progress_table.rows[0].renderable = Panel("Executing...", title="Status", border_style="green")
            jobs = await fetch_jobs()

            if jobs == "Unauthorized":
                console.log("[bold red]Unauthorized access. Please check your credentials.[/bold red]")
                break
            elif jobs:
                job_id = jobs[0]['id']
                console.log(f"Executing JobId: {job_id}")
                job_details_path = await fetch_and_save_job_details(job_id)
                if job_details_path:
                    console.log(f"[green]Job {job_id} fetched successfully[/green]")
                    progress_table.rows[0].renderable = Panel(f"Processing JobId: {job_id}", title="Status", border_style="yellow")

                    with open(job_details_path, 'r') as file:
                        job_details = json.load(file)
                    await process_job(
                        job_details,
                        run_on_runpod=args.runpod,
                        runpod_api_key=args.runpod_api_key
                    )

                    progress_table.rows[0].renderable = Panel("Executing...", title="Status", border_style="green")
                    console.log(f"[green]Job {job_id} processed successfully[/green]")
            await asyncio.sleep(2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated training and uploading")
    parser.add_argument('--wallet_address', type=str, required=True)
    parser.add_argument('--runpod', action='store_true', help="Run the job on RunPod")
    parser.add_argument('--runpod_api_key', type=str, help="RunPod API key")
    args = parser.parse_args()
    asyncio.run(main(args))
