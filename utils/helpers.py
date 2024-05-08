import asyncio
import json
import os
import sys

import aiohttp
from dotenv import load_dotenv

load_dotenv()
BASE_URL = os.getenv("BASE_URL")
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")

# current file directory
root = os.path.dirname(os.path.abspath(__file__))


async def get_token_and_miner_id():
    token_path = os.path.join(root, '../auth/auth', 'token.txt')
    miner_id_path = os.path.join(root, '../auth/auth', 'miner_id.txt')
    try:
        with open(token_path, 'r') as f:
            token = f.read().strip()
        with open(miner_id_path, 'r') as f:
            miner_id = f.read().strip()
        return token, miner_id
    except Exception as e:
        print(f"Error reading token or miner ID: {e}")
        sys.exit(1)


async def fetch_jobs():
    print("Waiting for training jobs")
    token, _ = await get_token_and_miner_id()
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {token}'}
        async with session.get(f"{BASE_URL}/pending-jobs", headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"Failed to fetch jobs: {await response.text()}")
                return []


async def fetch_and_save_job_details(job_id):
    token, miner_id = await get_token_and_miner_id()
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {token}'}
        async with session.post(f"{BASE_URL}/start-training/{job_id}", headers=headers,
                                json={'minerId': miner_id}) as response:
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
                print(f"Failed to start training for job {job_id}: {await response.text()}")
                return None


async def update_job_status(job_id, status):
    url = f"{BASE_URL}/update-status/{job_id}"
    token, _ = await get_token_and_miner_id()
    async with aiohttp.ClientSession() as session:
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        async with session.patch(url, json={'status': status}, headers=headers) as response:
            try:
                if response.status == 200:
                    print(f"Status updated to {status} for job {job_id}")
                else:
                    response.raise_for_status()
            except aiohttp.ClientResponseError as err:
                print(f"Failed to update status for job {job_id}: {err}")
            except Exception as e:
                print(f"An error occurred: {e}")
