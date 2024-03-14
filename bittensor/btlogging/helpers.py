import logging

def all_loggers():
    """
    Generator that yields all logger instances in the application.
    """
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.PlaceHolder):
            continue
        # In some versions of Python, the values in loggerDict might be
        # LoggerAdapter instances instead of Logger instances.
        # We check for Logger instances specifically.
        if isinstance(logger, logging.Logger):
            yield logger
        else:
            # If it's not a Logger instance, it could be a LoggerAdapter or
            # another form that doesn't directly offer logging methods.
            # This branch can be extended to handle such cases as needed.
            pass


def all_logger_names():
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.PlaceHolder):
            continue
        # In some versions of Python, the values in loggerDict might be
        # LoggerAdapter instances instead of Logger instances.
        # We check for Logger instances specifically.
        if isinstance(logger, logging.Logger):
            yield name
        else:
            # If it's not a Logger instance, it could be a LoggerAdapter or
            # another form that doesn't directly offer logging methods.
            # This branch can be extended to handle such cases as needed.
            pass


def get_max_logger_name_length():
    max_length = 0
    for name in all_logger_names():
        if len(name) > max_length:
            max_length = len(name)
    return max_length