import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_debug = "%(asctime)s - %(levelname)s - %(message)s"
    format_default = (
        "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {  # noqa: RUF012
        logging.DEBUG: format_debug + reset,
        logging.INFO: format_debug + reset,
        logging.WARNING: yellow + format_default + reset,
        logging.ERROR: red + format_default + reset,
        logging.CRITICAL: bold_red + format_default + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(log_to_file=False):
    # Create a logger
    logger = logging.getLogger("gear-cm")
    # logger = logging.getLogger("Gear-CM logger")
    logger.setLevel(logging.DEBUG)

    # logging.getLogger() returns the same logger object on every call, so
    # calling setup_logging() more than once in a process (re-running a
    # notebook cell, a hyperparameter-search loop, a retry) would otherwise
    # keep stacking handlers and duplicate every log line once per call.
    logger.handlers.clear()

    # Create a formatter to define the log format

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s   %(message)s", "%H:%M:%S")
    # console_handler.setFormatter(formatter)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    if log_to_file:
        # Make sure the logs directory exists
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create a file handler to write logs to a file
        # Separate each day into a different log file
        file_handler = TimedRotatingFileHandler(
            log_dir / "runs.log", when="midnight", interval=1, backupCount=7
        )
        file_handler.setLevel(logging.INFO)
        # formatter = logging.Formatter(
        #     "%(asctime)s - %(levelname)s   %(message)s", "%Y-%m-%d %H:%M:%S"
        # )
        # file_handler.setFormatter(formatter)
        file_handler.setFormatter(CustomFormatter())
        logger.addHandler(file_handler)

    return logger
