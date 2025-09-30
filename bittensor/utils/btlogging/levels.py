from bittensor.utils.btlogging import logging


# Logging level setup helpers.
def trace(on: bool = True):
    """
    Enables or disables trace logging.

    Parameters:
        on: If True, enables trace logging. If False, disables trace logging.
    """
    logging.set_trace(on)


def debug(on: bool = True):
    """
    Enables or disables debug logging.

    Parameters:
        on: If True, enables debug logging. If False, disables debug logging.
    """
    logging.set_debug(on)


def warning(on: bool = True):
    """
    Enables or disables warning logging.

    Parameters:
        on: If True, enables warning logging. If False, disables warning logging and sets default (WARNING) level.
    """
    logging.set_warning(on)


def info(on: bool = True):
    """
    Enables or disables info logging.

    Parameters:
        on: If True, enables info logging. If False, disables info logging and sets default (WARNING) level.
    """
    logging.set_info(on)
