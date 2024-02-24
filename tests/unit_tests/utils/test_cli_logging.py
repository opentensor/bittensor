import time
import re

from tests.helpers import MockConsole

console = MockConsole()


def escape_ansi(line):
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -\/]*[@-~]')
    return ansi_escape.sub('', line)


def test_console_success():
    console.success("Success message")
    assert console.captured_print == "✔ \x1b[32mSuccess message\x1b[0m\n"


def test_console_error(capsys):
    console.error("Error message")
    assert console.captured_print == "❌ \x1b[31mError message\x1b[0m\n"


def test_console_print(capsys):
    console.print("Random unformatted message")
    assert console.captured_print == "Random unformatted message"


def test_console_status(capsys):
    ending = "Finished waiting"
    with console.status("Waiting..."):
        time.sleep(2)
        console.success(ending)
        time.sleep(1)

    expected_length = len(ending)
    actual_ending = escape_ansi(
        console.captured_print
    ).strip()[-expected_length:]

    assert ending == actual_ending


def test_console_status_no_context(capsys):
    console.status("Waiting...")
    console.print("End")

    assert console.captured_print[-3:] == "End"


def test_console_status_multiline(capsys):
    with console.status("Waiting..."):
        time.sleep(2)
        console.print(
            "Balance:\n  <blue>{}</blue> \u27A1 <green>{}</green>\n".format(
                "τ0.000000002", "τ0.000000000"
            )
        )
        console.print(
            "Balance:\n  <blue>{}</blue> \u27A1 <green>{}</green>\n".format(
                "τ0.000000001", "τ0.000000000"
            )
        )

    with console.status("Waiting 2 ..."):
        time.sleep(2)

    console.print("End")

    assert escape_ansi(console.captured_print)[-3:] == "End"
