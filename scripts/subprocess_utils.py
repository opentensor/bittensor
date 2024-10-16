import os
import sqlite3
from typing import Optional

import subprocess
import psutil


def is_process_running(process_name: str) -> bool:
    """Check if a process with a given name is currently running."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and (process_name in proc.info['name'] or any(process_name in cmd for cmd in cmdline)):
            return True
    return False


def get_process(process_name: str) -> Optional[int]:
    """Check if a process with a given name is currently running, and return its PID if found."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and (process_name in proc.info['name'] or any(process_name in cmd for cmd in cmdline)):
            return proc.info['pid']
    return None


def start_commit_reveal_subprocess():
    """Start the commit reveal subprocess if not already running."""
    process_name = 'commit_reveal.py'
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "subprocess", "commit_reveal.py"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    if not is_process_running(process_name):
        stdout_file = open("/Users/daniel/repos/bittensor-sdk/scripts/subprocess/logs/commit_reveal_stdout.log", "w")
        stderr_file = open("/Users/daniel/repos/bittensor-sdk/scripts/subprocess/logs/commit_reveal_stderr.log", "w")
        print(f"Starting subprocess '{process_name}'...")
        env = os.environ.copy()
        env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

        process = subprocess.Popen(
            ['python3', script_path],
            stdout=stdout_file,
            stderr=stderr_file,
            preexec_fn=os.setsid,
            env=env
        )
        print(f"Subprocess '{process_name}' started with PID {process.pid}.")

        # Read and print what was captured to files
        with open("/Users/daniel/repos/bittensor-sdk/scripts/subprocess/logs/commit_reveal_stdout.log") as f:
            print("Subprocess output:")
            print(f.read())

        with open("/Users/daniel/repos/bittensor-sdk/scripts/subprocess/logs/commit_reveal_stderr.log") as f:
            print("Subprocess errors:")
            print(f.read())

    else:
        print(f"Subprocess '{process_name}' is already running.")


class DB:
    """
    For ease of interaction with the SQLite database used for --reuse-last and --html outputs of tables
    """

    def __init__(
        self,
        db_path: str = os.path.expanduser("~/.bittensor/bittensor.db"),
        row_factory=None,
    ):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self.row_factory = row_factory

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = self.row_factory
        return self.conn, self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()


def create_table(title: str, columns: list[tuple[str, str]], rows: list[list]) -> None:
    """
    Creates and populates the rows of a table in the SQLite database.

    :param title: title of the table
    :param columns: [(column name, column type), ...]
    :param rows: [[element, element, ...], ...]
    :return: None
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
    Reads a table from a SQLite database, returning back a column names and rows as a tuple
    :param table_name: the table name in the database
    :param order_by: the order of the columns in the table, optional
    :return: ([column names], [rows])
    """
    with DB() as (conn, cursor):
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        column_names = [info[1] for info in columns_info]
        column_types = [info[2] for info in columns_info]
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