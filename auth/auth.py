import json
import os
import sys
import requests
from dotenv import load_dotenv, set_key

# Load environment variables
load_dotenv()
BASE_URL = os.getenv("BASE_URL")

def authenticate(username, password):
    print("Username & password: ", username, " ", password)
    url = f"{BASE_URL}/login"
    response = requests.post(url, json={'username': username, 'password': password})
    if response.status_code == 200:
        token = response.json()['token']
        miner_id = response.json()['minerId']
        
        # Save token and miner ID to .env file
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        set_key(env_path, "TOKEN", token)
        set_key(env_path, "MINER_ID", str(miner_id))

        print("Authentication successful. Token and Miner ID saved.")
    else:
        print("Authentication failed.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python auth.py command [args]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "login" and len(sys.argv) == 4:
        username = sys.argv[2]
        password = sys.argv[3]
        authenticate(username, password)
    else:
        print("Invalid command or arguments")
