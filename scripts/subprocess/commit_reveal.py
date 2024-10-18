import argparse
import json
import os
import shlex
import sys
import time
import socket
import threading
from bittensor.core.subtensor import Subtensor
from bittensor_wallet import Wallet
from scripts import subprocess_utils as utils
from typing import List, Any

# Path to the SQLite database
DB_PATH = os.path.expanduser("~/.bittensor/bittensor.db")
# Global variable to control the server loop
running = True


class Commit:
    def __init__(self, wallet_hotkey_name: str, wallet_hotkey_ss58: str, wallet_name: str, wallet_path: str,
                 commit_hash: str, netuid: int, commit_block: int, reveal_block: int, uids: List[int],
                 weights: List[int], salt: List[int], version_key: int):
        self.wallet_hotkey_name = wallet_hotkey_name
        self.wallet_hotkey_ss58 = wallet_hotkey_ss58
        self.wallet_name = wallet_name
        self.wallet_path = wallet_path
        self.commit_hash = commit_hash
        self.netuid = netuid
        self.commit_block = commit_block
        self.reveal_block = reveal_block
        self.uids = uids
        self.weights = weights
        self.salt = salt
        self.version_key = version_key

    def to_dict(self) -> dict:
        return {
            "wallet_hotkey_name": self.wallet_hotkey_name,
            "wallet_hotkey_ss58": self.wallet_hotkey_ss58,
            "wallet_name": self.wallet_name,
            "wallet_path": self.wallet_path,
            "commit_hash": self.commit_hash,
            "netuid": self.netuid,
            "commit_block": self.commit_block,
            "reveal_block": self.reveal_block,
            "uids": json.dumps(self.uids),
            "weights": json.dumps(self.weights),
            "salt": json.dumps(self.salt),
            "version_key": self.version_key
        }

    @staticmethod
    def from_dict(data: dict) -> 'Commit':
        return Commit(
            wallet_hotkey_name=data["wallet_hotkey_name"],
            wallet_hotkey_ss58=data["wallet_hotkey_ss58"],
            wallet_name=data["wallet_name"],
            wallet_path=data["wallet_path"],
            commit_hash=data["commit_hash"],
            netuid=data["netuid"],
            commit_block=data["commit_block"],
            reveal_block=data["reveal_block"],
            uids=json.loads(data["uids"]),
            weights=json.loads(data["weights"]),
            salt=json.loads(data["salt"]),
            version_key=data["version_key"]
        )

    def __str__(self):
        return f"Commit(wallet_hotkey_name={self.wallet_hotkey_name}, wallet_hotkey_ss58={self.wallet_hotkey_ss58}, wallet_name={self.wallet_name}, wallet_path={self.wallet_path}, commit_hash={self.commit_hash}, netuid={self.netuid}, commit_block={self.commit_block}, reveal_block={self.reveal_block}, uids={self.uids}, weights={self.weights}, salt={self.salt}, version_key={self.version_key})"


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
        ("uids", "TEXT"),
        ("weights", "TEXT"),
        ("salt", "TEXT"),
        ("version_key", "INTEGER")
    ]
    # Check if the 'commits' table exists before creating it
    if not table_exists("commits"):
        print("Creating table 'commits'...")
        utils.create_table("commits", columns, [])
    else:
        print("Table 'commits' already exists.")


def reveal(subtensor, commit: Commit):
    # create wallet
    wallet = Wallet(name=commit.wallet_name, path=commit.wallet_path, hotkey=commit.wallet_hotkey_name)
    success, message = subtensor.reveal_weights(
        wallet=wallet,
        netuid=commit.netuid,
        uids=commit.uids,
        weights=commit.weights,
        salt=commit.salt,
        wait_for_inclusion=True,
        wait_for_finalization=True
    )
    # delete wallet object
    del wallet
    if success:
        print(f"Reveal success for commit {commit}")
    else:
        print(f"Reveal failure for commit: {message}")


def revealed(wallet_name, wallet_path, wallet_hotkey_str, wallet_hotkey_ss58, netuid, uids, weights, salt, version_key):
    try:
        # Check if a row with the specified data exists in the 'commits' table
        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            sql = (
                "SELECT COUNT(*) FROM commits WHERE wallet_hotkey_str=? AND  wallet_hotkey_ss58=? AND wallet_name=? AND wallet_path=? AND netuid=? AND "
                "uids=? AND weights=? AND salt=? AND version_key=?")
            cursor.execute(sql, (
                wallet_hotkey_str, wallet_hotkey_ss58, wallet_name, wallet_path, netuid, json.dumps(uids),
                json.dumps(weights),
                json.dumps(salt), version_key))
            count = cursor.fetchone()[0]
            if count > 0:
                # Delete the row if it exists
                delete_sql = (
                    "DELETE FROM commits WHERE wallet_hotkey_str=? AND  wallet_hotkey_ss58=? AND wallet_name=? AND wallet_path=? AND netuid=? AND "
                    "uids=? AND weights=? AND salt=? AND version_key=?")
                cursor.execute(delete_sql, (
                    wallet_hotkey_str, wallet_hotkey_ss58, wallet_name, wallet_path, netuid, json.dumps(uids),
                    json.dumps(weights), json.dumps(salt), version_key))
                conn.commit()
                print(
                    f"Deleted existing row with specified data: wallet_hotkey_str={wallet_hotkey_str}, wallet_hotkey_ss58={wallet_hotkey_ss58}, wallet_name={wallet_name}, wallet_path={wallet_path}, netuid={netuid}, uids={uids}, weights={weights}, salt={salt}, version_key={version_key}")
            else:
                print(
                    f"No existing row found with specified data: wallet_hotkey_str={wallet_hotkey_str}, wallet_hotkey_ss58={wallet_hotkey_ss58}, wallet_name={wallet_name}, wallet_path={wallet_path}, netuid={netuid}, uids={uids}, weights={weights}, salt={salt}, version_key={version_key}")
    except Exception as e:
        print(f"Error removing from table 'commits': {e}")


def revealed_hash(commit_hash: str):
    try:
        # Check if a row with the specified data exists in the 'commits' table
        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            sql = (
                "SELECT COUNT(*) FROM commits WHERE commit_hash=?")
            cursor.execute(sql, (commit_hash,))
            count = cursor.fetchone()[0]
            if count > 0:
                # Delete the row if it exists
                delete_sql = (
                    "DELETE FROM commits WHERE commit_hash=?")
                cursor.execute(delete_sql, (commit_hash,))
                conn.commit()
                print(f"\nDeleted existing row with commit hash {commit_hash}")
            else:
                print(f"\nNo existing row found with commit hash {commit_hash}")
    except Exception as e:
        print(f"Error removing from table 'commits': {e}")


def committed(commit: Commit):
    with utils.DB(db_path=DB_PATH) as (conn, cursor):
        commit_data = commit.to_dict()
        column_names = ", ".join(commit_data.keys())
        data = ", ".join(["?"] * len(commit_data))
        sql = f"INSERT INTO commits ({column_names}) VALUES ({data})"
        cursor.execute(sql, tuple(commit_data.values()))
        conn.commit()
    print(f"Committed commit data: {commit_data}")


def check_reveal(subtensor: Subtensor):
    try:
        columns, rows = utils.read_table("commits")
        commits = [Commit.from_dict(dict(zip(columns, commit))) for commit in rows]
    except Exception as e:
        print(f"Error reading table 'commits': {e}")
        return False

    if commits:
        # Sort commits by reveal block asc, and if two reveal blocks are the same, sort them by commit blocks asc
        commits.sort(key=lambda commit: (commit.reveal_block, commit.commit_block))
        next_reveal = commits[0]
        curr_block = subtensor.get_current_block()
        if next_reveal.reveal_block <= curr_block:
            reveal(subtensor, next_reveal)
            # # Delete the row after revealing, and delete all older reveals
            revealed_hash(next_reveal.commit_hash)
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
                # revealed "{wallet.name}" "{wallet.path}" "{wallet.hotkey_str}" "{wallet.hotkey.ss58_address}" "{netuid}" "{uids}" "{weights}" "{salt}" "{version_key}"
                revealed(args[1], args[2], args[3], args[4], args[5], json.loads(args[6]), json.loads(args[7]),
                         json.loads(args[8]), int(args[9]))
            elif command == 'revealed_hash':
                # revealed_hash "{commit_hash}"
                revealed_hash(args[1])
            elif command == 'committed':
                # wallet_name, wallet_path, wallet_hotkey_name, wallet_hotkey_ss58, curr_block, reveal_block, commit_hash, netuid, uids, weights, salt, version_key
                commit = Commit(
                    wallet_hotkey_name=args[3],
                    wallet_hotkey_ss58=args[4],
                    wallet_name=args[1],
                    wallet_path=args[2],
                    commit_hash=args[7],
                    netuid=int(args[8]),
                    commit_block=int(args[5]),
                    reveal_block=int(args[6]),
                    uids=json.loads(args[9]),
                    weights=json.loads(args[10]),
                    salt=json.loads(args[11]),
                    version_key=int(args[12])
                )
                committed(commit)
            elif command == 'terminate':
                terminate_process(None, None)
            else:
                print("Command not recognized")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()


def start_socket_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', 9949))
    server.listen(5)
    print('Listening on port 9949...')
    while running:
        client_sock, addr = server.accept()
        client_handler = threading.Thread(
            target=handle_client_connection,
            args=(client_sock,)
        )
        client_handler.start()


def terminate_process(signal_number, frame):
    global running
    print(f"Terminating process with signal {signal_number}")
    running = False
    sys.exit(0)


def main(args):
    # Initialize database and create table if necessary
    print("Initializing database...")
    initialize_db()
    subtensor = Subtensor(network=args.network, subprocess_initialization=False)  # Using network argument
    # A new block is created every 12 seconds. Check if the current block is equal to the reveal block
    server_thread = threading.Thread(target=start_socket_server)
    server_thread.start()
    while running:
        if check_reveal(subtensor=subtensor):
            print(f"Revealing commit for block {subtensor.get_current_block()}")
        else:
            print(f"Nothing to reveal for block {subtensor.get_current_block()}")
        time.sleep(args.sleep_interval)  # Using sleep interval argument


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Bittensor commit-reveal subprocess script.")
    parser.add_argument("--network", type=str, default="ws://localhost:9945", help="Subtensor network address")
    # TODO: have finney be default
    # parser.add_argument("--network", type=str, default="wss://entrypoint-finney.opentensor.ai:443", help="Subtensor network address")
    parser.add_argument("--sleep-interval", type=float, default=12, help="Interval between block checks in seconds")
    # Add more arguments as needed
    args = parser.parse_args()
    main(args)
