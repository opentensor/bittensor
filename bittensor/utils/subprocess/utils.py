import os
import re
import socket
import sqlite3
import subprocess
import time
from datetime import datetime
from typing import Optional
from bittensor.utils.btlogging import logging
import psutil

LOG_DIR = os.path.join(os.path.expanduser("~"), ".bittensor", "logs")
COMMIT_REVEAL_PROCESS = "commit_reveal.py"
PORT = 9949
HOST = "127.0.0.1"
# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)


def get_cr_log_files() -> tuple[str, str]:
    """
    Get the log files for the current running process.
    Returns:
        tuple[str, str]: Paths to the stdout log file and stderr log file.
    """
    pid = get_process(COMMIT_REVEAL_PROCESS)
    if pid is None:
        raise RuntimeError(f"Process '{COMMIT_REVEAL_PROCESS}' is not running.")

    # Define a regex pattern to match log files with timestamps
    log_pattern = re.compile(r"commit_reveal_(stdout|stderr)_(\d{8}_\d{6})\.log")

    stdout_log = None
    stderr_log = None
    latest_timestamp = None

    for log_file in os.listdir(LOG_DIR):
        match = log_pattern.match(log_file)
        if match:
            log_type, timestamp = match.groups()
            # Update latest log files if this file is more recent
            if latest_timestamp is None or timestamp > latest_timestamp:
                latest_timestamp = timestamp
                if log_type == "stdout":
                    stdout_log = os.path.join(LOG_DIR, log_file)
                elif log_type == "stderr":
                    stderr_log = os.path.join(LOG_DIR, log_file)

    if not (stdout_log and stderr_log):
        raise RuntimeError("Log files not found or incomplete.")

    return stdout_log, stderr_log


def is_process_running(process_name: str) -> bool:
    """
    Check if a process with a given name is currently running.

    Args:
        process_name (str): Name of the process to check.

    Returns:
        bool: True if the process is running, False otherwise.
    """
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        cmdline = proc.info["cmdline"]
        if cmdline and (
            process_name in proc.info["name"]
            or any(process_name in cmd for cmd in cmdline)
        ):
            return True
    return False


def get_process(process_name: str) -> Optional[int]:
    """
    Check if a process with a given name is currently running, and return its PID if found.

    Args:
        process_name (str): Name of the process to check.

    Returns:
        Optional[int]: PID of the process if found, None otherwise.
    """
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        cmdline = proc.info["cmdline"]
        if cmdline and (
            process_name in proc.info["name"]
            or any(process_name in cmd for cmd in cmdline)
        ):
            return proc.info["pid"]
    return None


def is_commit_reveal_subprocess_ready() -> bool:
    """
    Check the logs for the message 'commit_reveal subprocess is ready' and return True if it's found.

    Returns:
        bool: True if the message is found in the logs, False otherwise.
    """
    try:
        stdout_log, stderr_log = get_cr_log_files()
    except RuntimeError as e:
        logging.error(str(e))
        return False

    def check_message_in_log(file_path: str, message_: str) -> bool:
        """Check if a specific message is present in the log file."""
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                for line in file:
                    if message_ in line:
                        return True
        return False

    message = "commit_reveal subprocess is ready"
    return check_message_in_log(stdout_log, message)


def is_table_empty(table_name: str) -> bool:
    """
    Checks if a table in the database exists and is empty.

    Args:
        table_name (str): The name of the table to check.

    Returns:
        bool: True if the table does not exist or is empty, False otherwise.
    """
    try:
        with DB() as (conn, cursor):
            # Check if table exists
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
            )
            table_exists = cursor.fetchone()
            if not table_exists:
                logging.debug(f"Table '{table_name}' does not exist.")
                return True

            # Check if table is empty
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            if count == 0:
                logging.debug(f"Table '{table_name}' is empty.")
                return True
            else:
                logging.debug(f"Table '{table_name}' is not empty.")
                return False
    except Exception as e:
        logging.error(f"Error checking if table '{table_name}' is empty: {e}")
        return False


def start_if_existing_commits(
    network: Optional[str] = None, sleep_interval: Optional[float] = None
):
    # check if table is empty
    if not is_table_empty("commits"):
        # Stop then restart in case there are updates to the code
        stop_commit_reveal_subprocess()
        start_commit_reveal_subprocess(network, sleep_interval)
    else:
        logging.info(
            "Existing commits table is empty. Skipping starting commit reveal subprocess until a commit is there."
        )


def _is_port_available(port: int = PORT, host: str = HOST) -> bool:
    """Checks if the specified port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except socket.error:
            return False


def start_commit_reveal_subprocess(
    network: Optional[str] = None, sleep_interval: Optional[float] = None
):
    """
    Start the commit reveal subprocess if not already running.

    Args:
        network (Optional[str]): Network name if any, optional.
        sleep_interval (Optional[float]): Sleep interval if any, optional.
    """
    script_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "subprocess", "commit_reveal.py")
    )
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if not is_process_running(COMMIT_REVEAL_PROCESS):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Correctly construct the paths for STDOUT and STDERR log files
        stdout_log = os.path.join(LOG_DIR, f"commit_reveal_stdout_{current_time}.log")
        stderr_log = os.path.join(LOG_DIR, f"commit_reveal_stderr_{current_time}.log")

        if not _is_port_available():
            logging.error(f":cross_mark: <red>Error: Port {PORT} is busy.</red>")
            return

        os.makedirs(LOG_DIR, exist_ok=True)

        logging.info(f"Starting subprocess '{COMMIT_REVEAL_PROCESS}'...")
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

        args = ["python3", script_path]
        if network:
            args.extend(["--network", network])
        if sleep_interval:
            args.extend(["--sleep-interval", str(sleep_interval)])

        try:
            # Create a new subprocess
            process = subprocess.Popen(
                args=args,
                stdout=open(stdout_log, "a"),  # Redirect subprocess stdout to log file
                stderr=open(stderr_log, "a"),  # Redirect subprocess stderr to log file
                preexec_fn=os.setsid,
                env=env,
            )
            logging.info(f"Subprocess '{COMMIT_REVEAL_PROCESS}' started with PID {process.pid}.")

            attempt_count = 0
            while not is_commit_reveal_subprocess_ready() and attempt_count < 5:
                time.sleep(5)
                logging.debug(
                    f"Waiting for commit_reveal subprocess to be ready. Attempt {attempt_count + 1}..."
                )
                attempt_count += 1

            if attempt_count >= 5:
                logging.warning("Max start attempts reached. Subprocess may not be ready.")
        except Exception as e:
            logging.error(f"Failed to start subprocess '{COMMIT_REVEAL_PROCESS}': {e}")
    else:
        logging.error(f"Subprocess '{COMMIT_REVEAL_PROCESS}' is already running.")


def stop_commit_reveal_subprocess():
    """
    Stop the commit reveal subprocess if it is running.
    """
    pid = get_process(COMMIT_REVEAL_PROCESS)

    if pid is not None:
        logging.debug(f"Stopping subprocess '{COMMIT_REVEAL_PROCESS}' with PID {pid}...")
        os.kill(pid, 15)  # SIGTERM
        logging.debug(f"Subprocess '{COMMIT_REVEAL_PROCESS}' stopped.")
    else:
        logging.debug(f"Subprocess '{COMMIT_REVEAL_PROCESS}' is not running.")


class DB:
    """
    For ease of interaction with the SQLite database used for --reuse-last and --html outputs of tables
    """

    def __init__(
        self,
        db_path: str = os.path.join(os.path.expanduser("~"), ".bittensor", "bittensor.db"),
        row_factory=None,
    ):
        if not os.path.exists(os.path.dirname(db_path)):
            os.makedirs(os.path.dirname(db_path))

        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.row_factory = row_factory

    def __enter__(self) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = self.row_factory
        return self.conn, self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()


def create_table(title: str, columns: list[tuple[str, str]], rows: list[list]) -> bool:
    """
    Creates and populates the rows of a table in the SQLite database.

    Args:
        title (str): title of the table.
        columns (list[tuple[str, str]]): List of tuples where each tuple contains column name and column type.
        rows (list[list]): List of lists where each sublist contains elements representing a row.

    Returns:
        bool: True if the table creation is successful, False otherwise.
    """
    blob_cols = []
    for idx, (_, col_type) in enumerate(columns):
        if col_type == "BLOB":
            blob_cols.append(idx)
    if blob_cols:
        for row in rows:
            for idx in blob_cols:
                row[idx] = row[idx].to_bytes(row[idx].bit_length() + 7, byteorder="big")
    try:
        with DB() as (conn, cursor):
            drop_query = f"DROP TABLE IF EXISTS {title}"
            cursor.execute(drop_query)
            conn.commit()
            columns_ = ", ".join([" ".join(x) for x in columns])
            creation_query = f"CREATE TABLE IF NOT EXISTS {title} ({columns_})"
            cursor.execute(creation_query)
            conn.commit()
            query = f"INSERT INTO {title} ({', '.join([x[0] for x in columns])}) VALUES ({', '.join(['?'] * len(columns))})"
            cursor.executemany(query, rows)
            conn.commit()
        return True
    except Exception as e:
        logging.error(f"Error creating table '{title}': {e}")
        return False


def read_table(table_name: str, order_by: str = "") -> tuple[list, list]:
    """
    Reads a table from a SQLite database, returning back a column names and rows.

    Args:
        table_name (str): The table name in the database.
        order_by (str): The order of the columns in the table, optional.

    Returns:
        tuple[list, list]: A tuple containing a list of column names and a list of rows.
    """
    with DB() as (conn, cursor):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        column_names = []
        column_types = []
        for info in columns_info:
            try:
                column_names.append(info[1])
                column_types.append(info[2])
            except IndexError:
                logging.error(f"Error retrieving column info: {info}")

        cursor.execute(f"SELECT * FROM {table_name} {order_by}")
        rows = cursor.fetchall()
    blob_cols = []
    for idx, col_type in enumerate(column_types):
        if col_type == "BLOB":
            blob_cols.append(idx)
    if blob_cols:
        rows = [list(row) for row in rows]
        for row in rows:
            for idx in blob_cols:
                row[idx] = int.from_bytes(row[idx], byteorder="big")
    return column_names, rows


def delete_all_rows(table_name: str):
    """
    Deletes all rows from a table in the SQLite database.

    Args:
        table_name (str): The name of the table where all rows should be deleted.
    """
    with DB() as (conn, cursor):
        delete_query = f"DELETE FROM {table_name}"
        cursor.execute(delete_query)
        conn.commit()
