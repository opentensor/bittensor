import os
import sqlite3
from typing import Optional

import subprocess
import psutil

STDOUT_LOG = "/commit_reveal_stdout.log"
STDERR_LOG = "/commit_reveal_stderr.log"
PROCESS_NAME = "commit_reveal.py"


def is_process_running(process_name: str) -> bool:
    """
    Check if a process with a given name is currently running.

    Args:
        process_name (str): Name of the process to check.

    Returns:
        bool: True if the process is running, False otherwise.
    """
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and (process_name in proc.info['name'] or any(process_name in cmd for cmd in cmdline)):
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
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and (process_name in proc.info['name'] or any(process_name in cmd for cmd in cmdline)):
            return proc.info['pid']
    return None


def read_commit_reveal_logs():
    """
    Read and print the last 50 lines of logs from the log path.
    """
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "subprocess", "logs"))
    stdout_path = os.path.join(log_path, STDOUT_LOG.lstrip("/"))
    stderr_path = os.path.join(log_path, STDERR_LOG.lstrip("/"))

    def read_last_n_lines(file_path: str, n: int) -> list:
        """Reads the last N lines from a file."""
        with open(file_path, 'r') as file:
            return file.readlines()[-n:]

    if os.path.exists(stdout_path):
        print("----- STDOUT LOG -----")
        print(''.join(read_last_n_lines(stdout_path, 50)))
    else:
        print(f"STDOUT log file not found at {stdout_path}")

    if os.path.exists(stderr_path):
        print("----- STDERR LOG -----")
        print(''.join(read_last_n_lines(stderr_path, 50)))
    else:
        print(f"STDERR log file not found at {stderr_path}")


def start_commit_reveal_subprocess(network: Optional[str] = None, sleep_interval: Optional[float] = None):
    """
    Start the commit reveal subprocess if not already running.

    Args:
        network (Optional[str]): Network name if any, optional.
        sleep_interval (Optional[float]): Sleep interval if any, optional.
    """
    log_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "subprocess", "logs"))
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "subprocess", "commit_reveal.py"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if not is_process_running(PROCESS_NAME):
        stdout_file = open(log_path + STDOUT_LOG, "w")
        stderr_file = open(log_path + STDERR_LOG, "w")
        print(f"Starting subprocess '{PROCESS_NAME}'...")
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

        args = ['python3', script_path]
        if network:
            args.extend(['--network', network])
        if sleep_interval:
            args.extend(['--sleep-interval', str(sleep_interval)])

        process = subprocess.Popen(
            args=args,
            stdout=stdout_file,
            stderr=stderr_file,
            preexec_fn=os.setsid,
            env=env
        )
        print(f"Subprocess '{PROCESS_NAME}' started with PID {process.pid}.")

    else:
        print(f"Subprocess '{PROCESS_NAME}' is already running.")


def stop_commit_reveal_subprocess():
    """
     Stop the commit reveal subprocess if it is running.
     """
    pid = get_process(PROCESS_NAME)

    if pid is not None:
        print(f"Stopping subprocess '{PROCESS_NAME}' with PID {pid}...")
        os.kill(pid, 15)  # SIGTERM
        print(f"Subprocess '{PROCESS_NAME}' stopped.")
    else:
        print(f"Subprocess '{PROCESS_NAME}' is not running.")


class DB:
    """
    For ease of interaction with the SQLite database used for --reuse-last and --html outputs of tables
    """

    def __init__(
            self,
            db_path: str = os.path.expanduser("~/.bittensor/bittensor.db"),
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


def create_table(title: str, columns: list[tuple[str, str]], rows: list[list]):
    """
    Creates and populates the rows of a table in the SQLite database.

    Args:
        title (str): title of the table.
        columns (list[tuple[str, str]]): List of tuples where each tuple contains column name and column type.
        rows (list[list]): List of lists where each sublist contains elements representing a row.
    """
    blob_cols = []
    for idx, (_, col_type) in enumerate(columns):
        if col_type == "BLOB":
            blob_cols.append(idx)
    if blob_cols:
        for row in rows:
            for idx in blob_cols:
                row[idx] = row[idx].to_bytes(row[idx].bit_length() + 7, byteorder="big")
    with DB() as (conn, cursor):
        drop_query = f"DROP TABLE IF EXISTS {title}"
        cursor.execute(drop_query)
        conn.commit()
        columns_ = ", ".join([" ".join(x) for x in columns])
        creation_query = f"CREATE TABLE IF NOT EXISTS {title} ({columns_})"
        conn.commit()
        cursor.execute(creation_query)
        conn.committed()
        query = f"INSERT INTO {title} ({', '.join([x[0] for x in columns])}) VALUES ({', '.join(['?'] * len(columns))})"
        cursor.executemany(query, rows)
        conn.commit()
    return


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
                print(f"Error retrieving column info: {info}")

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
