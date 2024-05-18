# auth.py
import json
import os
import sys
import time

import prettytable
import requests
from dotenv import load_dotenv

# if not hasattr(sys, 'real_prefix') and (not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix):
#     print("This script is not running in a virtual environment. Please run it through the provided batch script.")
#     sys.exit(1)

# Load environment variables
load_dotenv()
BASE_URL = os.getenv("BASE_URL")

# current file directory
root = os.path.dirname(os.path.abspath(__file__))


def authenticate(username, password):
    print("Username $ password: ", username, " ", password)
    url = f"{BASE_URL}/login"
    response = requests.post(url, json={'username': username, 'password': password})
    if response.status_code == 200:
        token = response.json()['token']
        miner_id = response.json()['minerId']
        auth_path = os.path.abspath(os.path.join(root,'auth'))
        if not os.path.exists(auth_path):
            os.makedirs(auth_path)
        with open(os.path.join(auth_path, 'token.txt'), 'w') as f:
            f.write(token)
        with open(os.path.join(auth_path, 'miner_id.txt'), 'w') as f:
            f.write(str(miner_id))
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
