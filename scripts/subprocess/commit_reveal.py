import argparse
import json
import os
import shlex
import time
import utils  # Ensure this import works
import socket
import threading

from bittensor.core.subtensor import Subtensor
from bittensor_wallet import Wallet
from scripts import subprocess_utils as utils

# Path to the SQLite database
DB_PATH = os.path.expanduser("~/.bittensor/bittensor.db")


def table_exists(table_name: str) -> bool:
    try:
        columns, rows = utils.read_table(table_name)
        print(f"Table '{table_name}' exists with columns: {columns}")
        return True
    except Exception as e:
        print(f"Table '{table_name}' does not exist: {e}")
        return False


def is_table_empty(table_name: str) -> bool:
    try:
        # Attempt to read the table
        columns, rows = utils.read_table(table_name)

        # Check if the table is empty
        if not rows:
            print(f"Table '{table_name}' is empty.")
            return True
        else:
            print(f"Table '{table_name}' is not empty.")
            return False

    except Exception as e:
        print(f"Error checking if table '{table_name}' is empty: {e}")
        return False


def initialize_db():
    # Create 'commits' table if it doesn't exist
    columns = [
        ("wallet_hotkey_name", "TEXT"),
        ("wallet_hotkey_ss58", "TEXT"),
        ("wallet_path", "TEXT"),
        ("wallet_name", "TEXT"),
        ("commit_hash", "TEXT"),
        ("netuid", "INTEGER"),
        ("commit_block", "INTEGER"),
        ("reveal_block", "INTEGER"),
        ("uids", "TEXT"),  # Store list as a JSON string for simplicity
        ("weights", "TEXT"),  # Store list as a JSON string for simplicity
        ("salt", "TEXT"),  # Store list as a JSON string for simplicity
    ]

    # Check if the 'commits' table exists before creating it
    if not table_exists("commits"):
        print("Creating table 'commits'...")
        utils.create_table("commits", columns, [])
    else:
        print("Table 'commits' already exists.")


def reveal(subtensor, data):
    # create wallet
    wallet_name = data["wallet_name"]
    wallet_path = data["wallet_path"]
    wallet_hotkey_name = data["wallet_hotkey_name"]

    wallet = Wallet(name=wallet_name, path=wallet_path, hotkey=wallet_hotkey_name)

    print(f"the data: {data}")
    print(f"wallet: {wallet}")

    commit_hash = data["commit_hash"]
    uids = list(map(int, json.loads(data["uids"])))
    weights = list(map(int, json.loads(data["weights"])))
    netuid = data["netuid"]
    salt = list(map(int, json.loads(data["salt"])))

    print(f"commit_hash: {commit_hash}")
    print(f"uids: {uids}")
    print(f"weights: {weights}")
    print(f"netuid: {netuid}")
    print(f"salt: {salt}")

    # Calls subtensor.reveal_weights
    success, message = subtensor.reveal_weights(
        wallet=wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        salt=salt,
        wait_for_inclusion=True,
        wait_for_finalization=True
    )

    # delete wallet object
    del wallet

    if success:
        print("Reveal success")
    else:
        print(f"Reveal failure: {message}")


def revealed(wallet_name, wallet_path, wallet_hotkey, netuid, uids, weights, salt):
    # Check if a row with the specified data exists in the 'commits' table
    with utils.DB(db_path=DB_PATH) as (conn, cursor):
        sql = "SELECT COUNT(*) FROM commits WHERE wallet_hotkey=? AND wallet_name=? AND wallet_path=? AND netuid=? AND uids=? AND weights=? AND salt=?"
        cursor.execute(sql, (wallet_hotkey, wallet_name, wallet_path, netuid, json.dumps(uids), json.dumps(weights), json.dumps(salt)))
        count = cursor.fetchone()[0]
    
        if count > 0:
            # Delete the row if it exists
            delete_sql = "DELETE FROM commits WHERE wallet_hotkey=? AND wallet_name=? AND wallet_path=? AND netuid=? AND uids=? AND weights=? AND salt=?"
            cursor.execute(delete_sql,
                           (wallet_hotkey, wallet_name, wallet_path, netuid, json.dumps(uids), json.dumps(weights), json.dumps(salt)))
            conn.commit()
            print("Deleted existing row with specified data")
        else:
            print("No existing row found with specified data")


def committed(wallet_name, wallet_path, wallet_hotkey_name, wallet_hotkey_ss58, curr_block, reveal_block, commit_hash, netuid, uids, weights, salt):

    commit_data = {
        "wallet_hotkey_name": wallet_hotkey_name,
        "wallet_hotkey_ss58": wallet_hotkey_ss58,
        "wallet_name": wallet_name,
        "wallet_path": wallet_path,
        "commit_hash": commit_hash,
        "netuid": netuid,
        "commit_block": curr_block,
        "reveal_block": reveal_block,
        "uids": json.dumps(uids),
        "weights": json.dumps(weights),
        "salt": json.dumps(salt),
    }
    with utils.DB(db_path=DB_PATH) as (conn, cursor):
        column_names = ", ".join(commit_data.keys())
        data = ", ".join(["?"] * len(commit_data))
        sql = f"INSERT INTO commits ({column_names}) VALUES ({data})"
        cursor.execute(sql, tuple(commit_data.values()))
        conn.commit()
        
    print("Committed commit data: {}", commit_data)


def check_reveal(subtensor, curr_block: int):
    try:
        columns, rows = utils.read_table("commits")
    except Exception as e:
        print(f"Error reading table 'commits': {e}")
        return False

    curr_reveal = None
    for commit in rows:
        row_dict = dict(zip(columns, commit))
        if row_dict['reveal_block'] == curr_block:
            curr_reveal = row_dict
            break

    if curr_reveal:
        reveal(subtensor, curr_reveal)
        # Delete the row after revealing, and delete all older reveals
        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            cursor.execute('DELETE FROM commits WHERE reveal_block <= ?', (curr_block,))
            conn.commit()
        return True

    return False


def handle_client_connection(client_socket):
    try:
        while True:
            request = client_socket.recv(1024).decode()
            if not request:
                break
            args = shlex.split(request)
            command = args[0]
            if command == 'revealed':
                # wallet_name, wallet_path, wallet_hotkey, netuid, uids, weights, salt
                revealed(args[1], args[2], args[3], args[4], json.loads(args[5]), json.loads(args[6]), json.loads(args[7]))
            elif command == 'committed':
                # wallet_name, wallet_path, wallet_hotkey_name, wallet_hotkey_ss58, curr_block, reveal_block, commit_hash, netuid, uids, weights, salt
                committed(args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], json.loads(args[9]), json.loads(args[10]),
                          json.loads(args[11]))
            else:
                print("Command not recognized")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()


def start_socket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 9999))
    server.listen(5)
    print('Listening on port 9999...')
    while True:
        client_sock, addr = server.accept()
        client_handler = threading.Thread(
            target=handle_client_connection,
            args=(client_sock,)
        )
        client_handler.start()


def main(args):
    # Initialize database and create table if necessary
    print("Initializing database...")
    initialize_db()
    subtensor = Subtensor(network=args.network)  # Using network argument
    # A new block is created every 12 seconds. Check if the current block is equal to the reveal block

    server_thread = threading.Thread(target=start_socket_server)
    server_thread.start()
    while True:
        # get curr block
        curr_block = subtensor.get_current_block()
        if check_reveal(subtensor=subtensor, curr_block=curr_block):
            print(f"Revealing commit for block {curr_block}")
        else:
            print(f"Nothing to reveal for block {curr_block}")
        time.sleep(args.sleep_interval)  # Using sleep interval argument


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Bittensor commit-reveal subprocess script.")
    parser.add_argument("--network", type=str, default="ws://localhost:9945", help="Subtensor network address")
    parser.add_argument("--sleep_interval", type=float, default=0.25, help="Interval between block checks in seconds")
    # Add more arguments as needed
    args = parser.parse_args()
    main(args)
