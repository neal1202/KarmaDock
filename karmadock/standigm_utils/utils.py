#!usr/bin/env python

# -*- coding:utf-8 -*-

# @time : 2024/07/11 17:17
# @author : Hokyun Jeon
# @contact : neal1202@gmail.com

# Copyright (C) 2024 Standigm, Inc.
# All rights reserved.

import sys
from pathlib import Path

from loguru import logger
from loguru._logger import Logger


def set_logger(
    existing_logger: Logger = None,
    log_dir: str = None,
    project_name: str = "default",
    module_name: str = "default",
    func_name: str = "default",
    time_for_filename: bool = True,
    log_level: str = "INFO",
    format: str = "[{time:YYYY-MM-DD HH:mm:ss}][{level}][{message}]",
) -> Logger:
    """
    Configures the logger based on the provided parameters.

    Args:
        existing_logger (Logger, optional): An existing loguru logger instance to configure. If None, a new logger instance will be used.
        log_dir (str, optional): Directory to save log files. If None, logs to console only.
        project_name (str): Name of the project.
        module_name (str): Name of the module.
        func_name (str): Name of the function.
        time_for_filename (bool): Whether to include time in the log filename.
        log_level (str): Logging level.
        format (str): Log message format.

    Returns:
        Logger: Configured loguru logger instance.
    """
    if existing_logger is None:
        logger_instance = logger
    else:
        logger_instance = existing_logger

    # Reset logger
    logger_instance.remove()

    if log_dir is None:
        logger_instance.add(sink=sys.stdout, level=log_level, format=format)
    else:
        log_filename = f"{project_name}_{module_name}_{func_name}"
        log_filename += "_{time}.log" if time_for_filename else ".log"
        log_path = Path(log_dir) / log_filename

        logger_instance.add(log_path, level=log_level, format=format)

    return logger_instance
