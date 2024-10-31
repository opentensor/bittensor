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
from bittensor.utils import subprocess_utils as utils
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
        expire_block (int): The block number at which the commit will be expired.
        uids (List[int]): The list of UIDs.
        weights (List[int]): The list of weights.
        salt (List[int]): The salt used for the commit.
        version_key (int): The version key.
        revealed (bool): Whether the commit has been revealed.
    """

    def __init__(
            self,
            wallet_hotkey_name: str,
            wallet_hotkey_ss58: str,
            wallet_name: str,
            wallet_path: str,
            commit_hash: str,
            netuid: int,
            commit_block: int,
            reveal_block: int,
            expire_block: int,
            uids: List[int],
            weights: List[int],
            salt: List[int],
            version_key: int,
            revealed: bool = False,
    ):
        self.wallet_hotkey_name = wallet_hotkey_name
        self.wallet_hotkey_ss58 = wallet_hotkey_ss58
        self.wallet_name = wallet_name
        self.wallet_path = wallet_path
        self.commit_hash = commit_hash
        self.netuid = netuid
        self.commit_block = commit_block
        self.reveal_block = reveal_block
        self.expire_block = expire_block
        self.uids = uids
        self.weights = weights
        self.salt = salt
        self.version_key = version_key
        self.revealed = revealed

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
            "expire_block": self.expire_block,
            "uids": json.dumps(self.uids),
            "weights": json.dumps(self.weights),
            "salt": json.dumps(self.salt),
            "version_key": self.version_key,
            "revealed": self.revealed,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Commit":
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
            expire_block=data["expire_block"],
            uids=json.loads(data["uids"]),
            weights=json.loads(data["weights"]),
            salt=json.loads(data["salt"]),
            version_key=data["version_key"],
            revealed=data.get("revealed", False),  # Default to False if not present
        )

    def __str__(self) -> str:
        """
        Returns a string representation of the commit.

        Returns:
            str: String representation of the commit.
        """
        return (f"Commit(wallet_hotkey_name={self.wallet_hotkey_name}, wallet_hotkey_ss58={self.wallet_hotkey_ss58}, "
                f"wallet_name={self.wallet_name}, wallet_path={self.wallet_path}, commit_hash={self.commit_hash}, "
                f"netuid={self.netuid}, commit_block={self.commit_block}, reveal_block={self.reveal_block}, "
                f"expire_block={self.expire_block}, uids={self.uids}, weights={self.weights}, salt={self.salt}, "
                f"version_key={self.version_key}, revealed={self.revealed})")


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
        ("expire_block", "INTEGER"),
        ("uids", "TEXT"),
        ("weights", "TEXT"),
        ("salt", "TEXT"),
        ("version_key", "INTEGER"),
        ("revealed", "BOOLEAN DEFAULT FALSE"),
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
    wallet = Wallet(
        name=commit.wallet_name,
        path=commit.wallet_path,
        hotkey=commit.wallet_hotkey_name,
    )
    success, message = subtensor.reveal_weights(
        wallet=wallet,
        netuid=commit.netuid,
        uids=commit.uids,
        weights=commit.weights,
        salt=commit.salt,
        version_key=commit.version_key,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    del wallet
    if success:
        revealed_commit(commit.commit_hash)
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

    wallet = Wallet(
        name=commits[0].wallet_name,
        path=commits[0].wallet_path,
        hotkey=commits[0].wallet_hotkey_name,
    )
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
        wait_for_finalization=True,
    )
    del wallet

    if success:
        for commit in commits:
            revealed_commit(commit.commit_hash)
            print(f"Reveal success for batch commit: {commit}")
    else:
        print(f"Reveal failure for batch commits: {message}")


def sync_commit_data(matching_commit, commit_block, reveal_block, expire_block):
    """
    Sync the commit data with the given block details.

    Args:
        matching_commit (Commit): The local commit object to be synced.
        commit_block (int): The block at which the commit occurred.
        reveal_block (int): The block at which the commit was revealed.
        expire_block (int): The block at which the commit will expire.
    """
    try:
        matching_commit.commit_block = commit_block
        matching_commit.reveal_block = reveal_block
        matching_commit.expire_block = expire_block

        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            update_sql = """
                UPDATE commits
                SET commit_block=?, reveal_block=?, expire_block=?
                WHERE commit_hash=?
            """
            cursor.execute(update_sql,
                           (commit_block, reveal_block, expire_block, matching_commit.commit_hash)
                           )
            conn.commit()
        print(
            f"Updated commit {matching_commit.commit_hash} with commit_block={commit_block}, reveal_block={reveal_block}, expire_block={expire_block}")
    except Exception as e:
        print(f"Error updating commit data: {e}")


def chain_hash_sync(subtensor: Subtensor, current_block: int):
    """
    Perform a verification to check if the local reveal list is consistent with the chain.

    Args:
        current_block (int): The current block number.
        subtensor (Subtensor): The subtensor network object.
    """
    try:
        # Retrieve all commits from the local database
        local_commits = get_all_commits()
        # Filter commits to only those that are not revealed
        local_commits = [commit for commit in local_commits if not commit.revealed]
        chain_commits = []
        # Group commits by wallet_hotkey_ss58
        if local_commits:
            unique_combinations = list({(commit.wallet_hotkey_ss58, commit.netuid) for commit in local_commits})

            for combination in unique_combinations:
                ss58, netuid = combination
                try:
                    response = subtensor.query_module(
                        module="SubtensorModule",
                        name="WeightCommits",
                        params=[netuid, ss58],
                    )

                    if not response.value:
                        print(f"No commits found for {combination}")
                        continue

                    for commit_hash, commit_block, reveal_block, expire_block in response.value:
                        chain_commits.append(commit_hash)
                        if expire_block < current_block:
                            continue
                        if any(c.commit_hash == commit_hash for c in local_commits) and reveal_block <= current_block:
                            matching_commit = next(
                                (commit for commit in local_commits if commit.commit_hash == commit_hash),
                                None)
                            if matching_commit:
                                if commit_block != matching_commit.commit_block or reveal_block != matching_commit.reveal_block or expire_block != matching_commit.expire_block:
                                    sync_commit_data(matching_commit, commit_block, reveal_block, expire_block)
                            else:
                                print(f"Could not find matching commit for hash: {commit_hash}")
                except Exception as e:
                    print(f"Error during subtensor query chain sync: {e}")
    except Exception as e:
        print(f"Error during chain_hash_sync: {e}")


def delete_old_commits(current_block: int, offset: int):
    """
    Deletes rows in the database where the current block is greater than the expire_block.
    Prints each commit before deleting it.

    Args:
        offset (int): The expired block offset to delete expired commits.
        current_block (int): The current block number.
    """
    try:
        commits = get_all_commits()
        if not commits:
            print("No commits found in the database.")
            return

        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            for commit in commits:
                if current_block + offset < commit.expire_block:
                    delete_sql = "DELETE FROM commits WHERE commit_hash=?"
                    cursor.execute(delete_sql, (commit.commit_hash,))
                    conn.commit()
                    print(f"Current block: {current_block}. Deleting expired Commit: {commit}")
    except Exception as e:
        print(f"Error deleting expired commits: {e}")


def revealed_commit(commit_hash: str):
    """
    Handles the revealed_hash command by updating the revealed status on the corresponding commit from the database using the commit hash.

    Args:
        commit_hash (str): The commit hash.
    """
    try:
        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            sql = "SELECT COUNT(*) FROM commits WHERE commit_hash=?"
            cursor.execute(sql, (commit_hash,))
            count = cursor.fetchone()[0]
            if count > 0:
                # Update the revealed status in the database
                update_sql = "UPDATE commits SET revealed = ? WHERE commit_hash = ?"
                cursor.execute(update_sql, (True, commit_hash))
                conn.commit()
                print(f"\nUpdated revealed status on existing row with commit hash {commit_hash}")
            else:
                print(f"\nNo existing row found with commit hash {commit_hash}")
    except Exception as e:
        print(f"Error updating from table 'commits': {e}")


def revealed_commit_batch(commit_hashes: List[str]):
    """
    Handles the revealed_batch_hash command by removing the corresponding commits from the database using the commit hashes.

    Args:
        commit_hashes (List[str]): The list of commit hashes.
    """
    try:
        if not commit_hashes:
            return
        with utils.DB(db_path=DB_PATH) as (conn, cursor):
            for commit_hash in commit_hashes:
                sql = "SELECT COUNT(*) FROM commits WHERE commit_hash=?"
                cursor.execute(sql, (commit_hash,))
                count = cursor.fetchone()[0]
                if count > 0:
                    update_sql = "UPDATE commits SET revealed = ? WHERE commit_hash = ?"
                    cursor.execute(update_sql, (True, commit_hash))
                    conn.commit()
                    print(f"\nUpdated revealed status on existing row with commit hash {commit_hash}")
                else:
                    print(f"\nNo existing row found with commit hash {commit_hash}")
    except Exception as e:
        print(f"Error updating from table 'commits': {e}")


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
    print(f"Committed data: {commit_data}")


def get_all_commits() -> List[Commit]:
    """
    Retrieves all commits from the database.

    Returns:
        List[Commit]: A list of all commits in the database.
    """
    columns, rows = utils.read_table("commits")
    return [Commit.from_dict(dict(zip(columns, commit))) for commit in rows]


def check_reveal(current_block: int) -> bool:
    """
    Checks if there are any commits to reveal.

    Args:
        current_block(int): The current block number.

    Returns:
        bool: True if a commit was revealed, False otherwise.
    """
    try:
        commits = get_all_commits()
        commits = [commit for commit in commits if not commit.revealed]

    except Exception as e:
        print(f"Error reading table 'commits': {e}")
        return False

    if commits:
        # Filter for commits that are ready to be revealed
        reveal_candidates = [
            commit for commit in commits if commit.reveal_block <= current_block <= commit.expire_block
        ]
        return len(reveal_candidates) > 0
    return False


def reveal_commits(subtensor: Subtensor, current_block: int):
    """
    Performs reveal on commits that are ready to be revealed.

    Args:
        current_block(int): The current block number.
        subtensor (Subtensor): The subtensor network object.
    """
    try:
        local_commits = get_all_commits()
        local_commits = [commit for commit in local_commits if not commit.revealed]
        local_reveals = [
            commit for commit in local_commits if commit.reveal_block <= current_block <= commit.expire_block
        ]
        chain_reveals = []
        if local_reveals:
            unique_combinations = list({(commit.wallet_hotkey_ss58, commit.netuid) for commit in local_reveals})
            # Dict that has ss58 as key, and latest commit block as value
            commit_dict: Dict[tuple[str, int], int] = {}
            for combination in unique_combinations:
                ss58, netuid = combination
                ready_to_reveal = []
                try:
                    response = subtensor.query_module(
                        module="SubtensorModule",
                        name="WeightCommits",
                        params=[netuid, ss58],
                    )

                    if not response.value:
                        print(f"No commits found for {combination}")
                        continue

                    for commit_hash, commit_block, reveal_block, expire_block in response.value:
                        if expire_block < current_block:
                            print(f"Commit {commit_hash} is expired.")
                            continue
                        if any(c.commit_hash == commit_hash for c in
                               local_reveals) and reveal_block <= current_block <= expire_block:
                            matching_commit = next(
                                (commit for commit in local_commits if commit.commit_hash == commit_hash),
                                None)
                            if matching_commit:
                                ready_to_reveal.append(matching_commit)
                            else:
                                print(f"Could not find commit hash {commit_hash} locally.")

                        if commit_block > commit_dict.get(combination, 0):
                            commit_dict[combination] = commit_block

                    if len(ready_to_reveal) > 1:
                        chain_reveals.extend(ready_to_reveal)
                        reveal_batch(subtensor, ready_to_reveal)
                    elif len(ready_to_reveal) == 1:
                        chain_reveals.extend(ready_to_reveal)
                        reveal(subtensor, ready_to_reveal[0])
                except Exception as e:
                    print(f"Error querying expected hashes for {combination}: {e}")

            # Compare reveal candidates and ready_to_reveal
            if set(chain_reveals) != set(local_reveals):  # there are left over local reveals
                print("there is a difference between local commits and chain commits")
                # Filter commits that are older than the newest one in commit_dict that was revealed
                for (ss58, netuid), newest_commit_block in commit_dict.items():
                    for commit in local_reveals:
                        if commit.wallet_hotkey_ss58 == ss58 and commit.netuid == netuid and commit.commit_block <= newest_commit_block:
                            # Mark the commit as revealed, as a newer commit as already been revealed
                            print(f"revealing commit {commit.commit_hash} as a newer hash was submitted")
                            commit.revealed = True
                            revealed_commit(commit.commit_hash)

    except Exception as e:
        print(f"Error reading table 'commits': {e}")


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
            if request.startswith('revealed_hash_batch'):
                try:
                    command = 'revealed_hash_batch'
                    json_start_index = request.index('[')
                    json_payload = request[json_start_index:]
                    args = json.loads(json_payload)
                    revealed_commit_batch(args)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON for {command}: {e}")
                except Exception as e:
                    print(f"Error processing {command}: {e}")
            else:
                args = shlex.split(request)
                command = args[0]
                commands = {
                    "revealed_hash": lambda: revealed_commit(args[1]),
                    "revealed_hash_batch": lambda: revealed_commit_batch(json.loads(args[1])),
                    "committed": lambda: committed(
                        Commit(
                            wallet_hotkey_name=args[3],
                            wallet_hotkey_ss58=args[4],
                            wallet_name=args[1],
                            wallet_path=args[2],
                            commit_hash=args[8],
                            netuid=int(args[9]),
                            commit_block=int(args[5]),
                            reveal_block=int(args[6]),
                            expire_block=int(args[7]),
                            uids=json.loads(args[10]),
                            weights=json.loads(args[11]),
                            salt=json.loads(args[12]),
                            version_key=int(args[13]),
                        )
                    ),
                    "terminate": lambda: terminate_process(None, None),
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
    server.bind(("127.0.0.1", 9949))
    server.listen(5)
    server.settimeout(2)  # Set timeout for any incoming requests to 2 seconds
    print("Listening on port 9949...")

    with ThreadPoolExecutor(max_workers=10) as executor:  # limit of workers amount
        while running:
            try:
                client_sock, addr = server.accept()
                print(f"Accepted connection from {addr[0]}.")
                executor.submit(handle_client_connection, client_sock)
            except socket.timeout:
                continue
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
    print(f"initializing subtensor with network: {args.network} and sleep time: {args.sleep_interval} seconds")
    subtensor = Subtensor(network=args.network, subprocess_initialization=False)
    server_thread = threading.Thread(target=start_socket_server)
    server_thread.start()

    counter = 0  # Initialize counter
    print("commit_reveal subprocess is ready")
    while running:
        counter += 1
        curr_block = subtensor.get_current_block()
        if check_reveal(curr_block):
            print(f"Revealing commit on block {curr_block}")
            reveal_commits(subtensor=subtensor, current_block=curr_block)

        if counter % 100 == 0:
            chain_hash_sync(subtensor=subtensor, current_block=curr_block)
            delete_old_commits(current_block=curr_block, offset=1000)

        time.sleep(args.sleep_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Bittensor commit-reveal subprocess script."
    )
    parser.add_argument(
        "--network",
        type=str,
        default="wss://entrypoint-finney.opentensor.ai:443",
        help="Subtensor network address",
    )
    parser.add_argument(
        "--sleep-interval",
        type=float,
        default=12.0,
        help="Interval between block checks in seconds",
    )
    args = parser.parse_args()
    main(args)
