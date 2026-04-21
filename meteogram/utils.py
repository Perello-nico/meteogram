# %%
import logging
import os
import drops2

# %% ###################################################
# LOGGING
########################################################

def resolve_log_level(verbosity: str | int | None) -> int:
    if isinstance(verbosity, int):
        return verbosity

    if verbosity is None:
        return logging.INFO

    verbosity_map = {
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
    }
    return verbosity_map.get(str(verbosity).lower(), logging.INFO)


def setup_logger(
    log_path: str | None = None,
    verbosity: str | int | None = 'info'
) -> logging.Logger:
    logger = logging.getLogger(__name__)
    level = resolve_log_level(verbosity)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s'
    )

    if not any(isinstance(handler, logging.StreamHandler) and
               not isinstance(handler, logging.FileHandler)
               for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
                handler.setFormatter(formatter)

    file_handlers = [
        handler for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]

    for handler in file_handlers:
        logger.removeHandler(handler)
        handler.close()

    if log_path is not None:
        log_path_abs = os.path.abspath(log_path)
        file_handler = logging.FileHandler(log_path_abs, mode='w', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# %% ###################################################
# DROPS2 UTILS
########################################################

def open_drops_door(url: str, user: str, password: str) -> None:
    drops2.set_credentials(url, user, password)