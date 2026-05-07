# %%
import logging
import os
import numpy.typing as npt
import numpy as np
from typing import Any, Literal
from .settings import LOGGER

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
    try:
        import drops2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "drops2 is required for DDS authentication. Install/configure drops2 "
            "before calling open_drops_door."
        ) from exc
    drops2.set_credentials(url, user, password)



# %% ###################################################
# DERIVATES
########################################################


def from_K_to_C(t: npt.NDArray) -> npt.NDArray:
    return t - 273.15


def compute_wind_speed(u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
    wind_speed = np.sqrt(u**2 + v**2)
    wind_direction = np.arctan2(v, u) * 180 / np.pi + 180
    return wind_speed


def compute_wind_direction(u: npt.NDArray, v: npt.NDArray) -> npt.NDArray:
    wind_direction = np.arctan2(v, u) * 180 / np.pi + 180
    return wind_direction


Derivates = Literal[
    "from_K_to_C",
    "compute_wind_speed",
    "compute_wind_direction",
]


def get_derivates_fn(
    function_code: Derivates,
    logger: logging.Logger | None = None
) -> Any:
    logger = logger or LOGGER

    match function_code:
        case "from_K_to_C":
            return from_K_to_C
        case "compute_wind_speed":
            return compute_wind_speed
        case "compute_wind_direction":
            return compute_wind_direction

    logger.error(f"Unknown function for computing derivates: {function_code!r}")

# %%
