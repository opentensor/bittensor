import argparse
import json
import os
import shlex
import sys
import time
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

from bittensor.core.subtensor import Subtensor
from bittensor_wallet import Wallet
from scripts import subprocess_utils as utils
from typing import List, Any, Dict, Optional

# Path to the SQLite database
DB_PATH = os.path.expanduser("~/.bittensor/bittensor.db")
# Global variable to control the server loop
running = True


class Commit:
    """
    A class representing a commit in the Bittensor network.

    Attributes:
        wallet_hotkey_name (str): The hotkey name associated with the wallet.
        wallet_hotkey_ss58 (str): The wallet's SS58 address.
        wallet_name (str): The wallet name.
        wallet_path (str): The path to the wallet.
        commit_hash (str): The commit hash.
        netuid (int): The network UID.
        commit_block (int): The block number at which the commit was made.
        reveal_block (int): The block number at which the commit will be revealed.
        uids (List[int]): The list of UIDs.
        weights (List[int]): The list of weights.
        salt (List[int]): The salt used for the commit.
        version_key (int): The version key.
    """

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

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the commit object to a dictionary.

        Returns:
            Dict[str, Any]: A dictionary representation of the commit.
        """
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
    def from_dict(data: Dict[str, Any]) -> 'Commit':
        """
        Creates a Commit object from a dictionary.

        Args:
            data (Dict[str, Any]): A dictionary containing commit data.

        Returns:
            Commit: A Commit object.
        """
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

    def __str__(self) -> str:
        """
        Returns a string representation of the commit.

        Returns:
            str: String representation of the commit.
        """
        return f"Commit(wallet_hotkey_name={self.wallet_hotkey_name}, wallet_hotkey_ss58={self.wallet_hotkey_ss58}, wallet_name={self.wallet_name}, wallet_path={self.wallet_path}, commit_hash={self.commit_hash}, netuid={self.netuid}, commit_block={self.commit_block}, reveal_block={self.reveal_block}, uids={self.uids}, weights={self.weights}, salt={self.salt}, version_key={self.version_key})"


def table_exists(table_name: str) -> bool:
    """
    Checks if a table exists in the database.

    Args:
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table exists, False otherwise.
    """
    try:
        columns, rows = utils.read_table(table_name)
        print(f"Table '{table_name}' exists with columns: {columns}")
        return True
    except Exception as e:
        print(f"Table '{table_name}' does not exist: {e}")
        return False


def is_table_empty(table_name: str) -> bool:
    """
    Checks if a table in the database is empty.

    Args:
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table is empty, False otherwise.
    """
    try:
        columns, rows = utils.read_table(table_name)
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
    """
    Initializes the database by creating the 'commits' table if it does not exist.
    """
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
    if not table_exists("commits"):
        print("Creating table 'commits'...")
        utils.create_table("commits", columns, [])
    else:
        print("Table 'commits' already exists.")


def reveal(subtensor: Subtensor, commit: Commit):
    """
    Reveals the weights for a commit to the subtensor network.

    Args:
        subtensor (Subtensor): The subtensor network object.
        commit (Commit): The commit object containing the data to be revealed.
    """
    wallet = Wallet(name=commit.wallet_name, path=commit.wallet_path, hotkey=commit.wallet_hotkey_name)
    success, message = subtensor.reveal_weights(
        wallet=wallet,
        netuid=commit.netuid,
        uids=commit.uids,
        weights=commit.weights,
        salt=commit.salt,
        version_key=commit.version_key,
        wait_for_inclusion=True,
        wait_for_finalization=True
    )
    del wallet
    if success:
        revealed_hash(commit.commit_hash)
        print(f"Reveal success for commit {commit}")
    else:
        print(f"Reveal failure for commit: {message}")


def reveal_batch(subtensor: Subtensor, commits: List[Commit]):
    """
    Reveals the weights for a batch of commits to the subtensor network.

    Args:
        subtensor (Subtensor): The subtensor network object.
        commits (List[Commit]): A list of commit objects to be revealed.
    """
    if not commits:
        print("reveal_batch has no commits to reveal.")
        return

    wallet = Wallet(name=commits[0].wallet_name, path=commits[0].wallet_path, hotkey=commits[0].wallet_hotkey_name)
    netuid = commits[0].netuid
    uids = [commit.uids for commit in commits]
    weights = [commit.weights for commit in commits]
    salt = [commit.salt for commit in commits]
    version_keys = [commit.version_key for commit in commits]

    success, message = subtensor.batch_reveal_weights(
        wallet=wallet,
        netuid=netuid,
        uids=uids,
        weights=weights,
        salt=salt,
        version_keys=version_keys,
        wait_for_inclusion=True,
        wait_for_finalization=True
    )
    del wallet

    if success:
        for commit in commits:
            revealed_hash(commit.commit_hash)
            print(f"Reveal success for batch commit: {commit}")
    else:
        print(f"Reveal failure for batch commits: {message}")


def chain_hash_check(subtensor: Subtensor):
    """
    Perform a verification to check if the local reveal list is consistent with the chain.

    Args:
        subtensor (Subtensor): The subtensor network object.
    """
    try:
        # Retrieve all commits from the local database
        commits = get_all_commits()

        # Group commits by wallet_hotkey_ss58
        commits_by_ss58 = {}
        for commit in commits:
            ss58 = commit.wallet_hotkey_ss58
            if ss58 not in commits_by_ss58:
                commits_by_ss58[ss58] = []
            commits_by_ss58[ss58].append(commit)

        if commits_by_ss58:
            for ss58, ss58_commits in commits_by_ss58.items():
                # Get a set of unique netuids from the commits
                netuids = set(commit.netuid for commit in ss58_commits)
                for netuid in netuids:
                    # Query the subtensor backend for commit hashes and blocks
                    response = subtensor.query_module(
                        module="SubtensorModule",
                        name="WeightCommits",
                        params=[netuid, ss58],
                    )

                    for commit_hash, commit_block in response.value:
                        # Check if any commit on Subtensor is absent in local database
                        if not any(c.commit_hash == commit_hash for c in ss58_commits):
                            print(f"There is a commit on Subtensor (hash: {commit_hash}) that we don't have locally.")

                    # Check if any local commit is absent in Subtensor
                    local_commit_hashes = {c.commit_hash for c in ss58_commits}
                    subtensor_commit_hashes = {commit_hash for commit_hash, _ in response.value}

                    for local_commit_hash in local_commit_hashes:
                        if local_commit_hash not in subtensor_commit_hashes:
                            print(f"There is a local commit (hash: {local_commit_hash}) that is not on Subtensor.")
                            revealed_hash(local_commit_hash)
    except Exception as e:
        print(f"Error during chain_hash_check: {e}")


def revealed(wallet_name: str, wallet_path: str, wallet_hotkey_str: str, wallet_hotkey_ss58: str, netuid: int,
             uids: List[int], weights: List[int], salt: List[int], version_key: int):
    """
    Handles the revealed command by removing the corresponding commit from the database.

    Args:
        wallet_name (str): The wallet name.
        wallet_path (str): The path to the wallet.
        wallet_hotkey_str (str): The wallet hotkey as a string.
        wallet_hotkey_ss58 (str): The wallet hotkey SS58 address.
        netuid (int): The network UID.
        uids (List[int]): The list of UIDs.
        weights (List[int]): The list of weights.
        salt (List[int]): The salt used for the commit.
        version_key (int): The version key.
    """
    try:
        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            sql = (
                "SELECT COUNT(*) FROM commits WHERE wallet_hotkey_str=? AND wallet_hotkey_ss58=? AND wallet_name=? AND wallet_path=? AND netuid=? AND "
                "uids=? AND weights=? AND salt=? AND version_key=?")
            cursor.execute(sql, (
                wallet_hotkey_str, wallet_hotkey_ss58, wallet_name, wallet_path, netuid, json.dumps(uids),
                json.dumps(weights),
                json.dumps(salt), version_key))
            count = cursor.fetchone()[0]
            if count > 0:
                delete_sql = (
                    "DELETE FROM commits WHERE wallet_hotkey_str=? AND wallet_hotkey_ss58=? AND wallet_name=? AND wallet_path=? AND netuid=? AND "
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
    """
    Handles the revealed_hash command by removing the corresponding commit from the database using the commit hash.

    Args:
        commit_hash (str): The commit hash.
    """
    try:
        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            sql = "SELECT COUNT(*) FROM commits WHERE commit_hash=?"
            cursor.execute(sql, (commit_hash,))
            count = cursor.fetchone()[0]
            if count > 0:
                delete_sql = "DELETE FROM commits WHERE commit_hash=?"
                cursor.execute(delete_sql, (commit_hash,))
                conn.commit()
                print(f"\nDeleted existing row with commit hash {commit_hash}")
            else:
                print(f"\nNo existing row found with commit hash {commit_hash}")
    except Exception as e:
        print(f"Error removing from table 'commits': {e}")


def revealed_batch_hash(commit_hashes: List[str]):
    """
    Handles the revealed_batch_hash command by removing the corresponding commits from the database using the commit hashes.

    Args:
        commit_hashes (List[str]): The list of commit hashes.
    """
    try:
        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            for commit_hash in commit_hashes:
                sql = "SELECT COUNT(*) FROM commits WHERE commit_hash=?"
                cursor.execute(sql, (commit_hash,))
                count = cursor.fetchone()[0]
                if count > 0:
                    delete_sql = "DELETE FROM commits WHERE commit_hash=?"
                    cursor.execute(delete_sql, (commit_hash,))
                    conn.commit()
                    print(f"\nDeleted existing row with commit hash {commit_hash}")
                else:
                    print(f"\nNo existing row found with commit hash {commit_hash}")
    except Exception as e:
        print(f"Error removing from table 'commits': {e}")


def committed(commit: Commit):
    """
    Commits a new commit object to the database.

    Args:
        commit (Commit): The commit object to save.
    """
    with utils.DB(db_path=DB_PATH) as (conn, cursor):
        commit_data = commit.to_dict()
        column_names = ", ".join(commit_data.keys())
        data = ", ".join(["?"] * len(commit_data))
        sql = f"INSERT INTO commits ({column_names}) VALUES ({data})"
        cursor.execute(sql, tuple(commit_data.values()))
        conn.commit()
    print(f"Committed commit data: {commit_data}")


def get_all_commits() -> List[Commit]:
    """
    Retrieves all commits from the database.

    Returns:
        List[Commit]: A list of all commits in the database.
    """
    columns, rows = utils.read_table("commits")
    return [Commit.from_dict(dict(zip(columns, commit))) for commit in rows]


def check_reveal(subtensor: Subtensor) -> bool:
    """
    Checks if there are any commits to reveal and performs the reveal if necessary.

    Args:
        subtensor (Subtensor): The subtensor network object.

    Returns:
        bool: True if a commit was revealed, False otherwise.
    """
    try:
        commits = get_all_commits()
    except Exception as e:
        print(f"Error reading table 'commits': {e}")
        return False

    if commits:
        curr_block = subtensor.get_current_block()

        # Filter for commits that are ready to be revealed
        reveal_candidates = [commit for commit in commits if commit.reveal_block <= curr_block]
        return len(reveal_candidates) > 0
    return False


def reveal_candidates(subtensor: Subtensor):
    """
    Performs reveal on commits that are ready to be revealed.

    Args:
        subtensor (Subtensor): The subtensor network object.
    """
    try:
        commits = get_all_commits()
    except Exception as e:
        print(f"Error reading table 'commits': {e}")

    if commits:
        curr_block = subtensor.get_current_block()

        # Filter for commits that are ready to be revealed
        ready_for_reveal = [commit for commit in commits if commit.reveal_block <= curr_block]
        if ready_for_reveal:
            # Group commits by wallet_hotkey_ss58
            grouped_reveals = {}
            for commit in ready_for_reveal:
                key = commit.wallet_hotkey_ss58
                if key not in grouped_reveals:
                    grouped_reveals[key] = []
                grouped_reveals[key].append(commit)

            # Process each group separately
            for hotkey_ss58, group in grouped_reveals.items():
                if len(group) > 1:
                    # Batch reveal if there are 2 or more reveal candidates
                    reveal_batch(subtensor, group)
                else:
                    # Otherwise, reveal individually
                    reveal(subtensor, group[0])


def handle_client_connection(client_socket: socket.socket):
    """
    Handles incoming client connections for the socket server.
    Args:
        client_socket (socket.socket): The client socket connection.
    """
    try:
        while True:
            request = client_socket.recv(1024).decode()
            if not request:
                break
            args = shlex.split(request)
            command = args[0]
            commands = {
                'revealed': lambda: revealed(
                    args[1], args[2], args[3], args[4], int(args[5]),
                    json.loads(args[6]), json.loads(args[7]),
                    json.loads(args[8]), int(args[9])
                ),
                'revealed_hash': lambda: revealed_hash(args[1]),
                'committed': lambda: committed(Commit(
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
                )),
                'terminate': lambda: terminate_process(None, None)
            }
            if command in commands:
                try:
                    commands[command]()
                except (IndexError, ValueError, json.JSONDecodeError) as e:
                    print(f"Error in processing command {command}: {e}")
            else:
                print(f"Command not recognized: {command}")
    except socket.error as e:
        print(f"Socket error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()


def start_socket_server():
    """
    Starts the socket server to listen for incoming connections.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', 9949))
    server.listen(5)
    server.settimeout(2) # Set timeout for any incoming requests to 2 seconds
    print('Listening on port 9949...')

    with ThreadPoolExecutor(max_workers=10) as executor:  # limit of workers amount
        while running:
            try:
                client_sock, addr = server.accept()
                print(f"Accepted connection from {addr[0]}.")
                executor.submit(handle_client_connection, client_sock)
            except socket.timeout:
                print("Socket timeout, continuing to listen...")
            except Exception as e:
                print(f"Error accepting connection: {e}.")
                break


def terminate_process(signal_number: Optional[int], frame: Optional[Any]):
    """
    Terminates the process gracefully.

    Args:
        signal_number (Optional[int]): The signal number causing the termination.
        frame (Optional[Any]): The current stack frame.
    """
    global running
    print(f"Terminating process with signal {signal_number} and/or frame {frame}")
    running = False
    sys.exit(0)


def main(args: argparse.Namespace):
    """
    The main function to run the Bittensor commit-reveal subprocess script.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """
    initialize_db()
    subtensor = Subtensor(network=args.network, subprocess_initialization=False)
    server_thread = threading.Thread(target=start_socket_server)
    server_thread.start()

    counter = 0  # Initialize counter

    while running:
        counter += 1

        if check_reveal(subtensor=subtensor):
            reveal_candidates(subtensor=subtensor)
            print(f"Revealing commit on block {subtensor.get_current_block()}")

        # Every 100th run, perform an additional check to verify reveal list alignment with the backend
        if counter % 100 == 0:
            chain_hash_check(subtensor=subtensor)

        time.sleep(args.sleep_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Bittensor commit-reveal subprocess script.")
    parser.add_argument("--network", type=str, default="wss://entrypoint-finney.opentensor.ai:443",
                        help="Subtensor network address")
    parser.add_argument("--sleep-interval", type=float, default=12, help="Interval between block checks in seconds")
    args = parser.parse_args()
    main(args)
