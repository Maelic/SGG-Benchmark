# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

current_step = 0
use_step = False

def logger_step(logger, info):
    global use_step
    if not use_step:
        return
    global current_step
    current_step += 1
    logger.info('#'*20+' Step '+str(current_step)+': '+info+' '+'#'*20)

def setup_logger(name, save_dir=None, distributed_rank=0, filename="log.txt", steps=False, verbose="INFO"):
    global use_step
    use_step = steps
    try:
        from loguru import logger
        assert verbose in ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], "Invalid verbose option: "+verbose
        level = verbose
        logger.remove()
        if distributed_rank is not None and distributed_rank > 0:
            return logger
        if save_dir is not None:
            logger.add(os.path.join(save_dir, filename),  format='{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}{function}{line} - <level>{message}</level>', colorize=False, level=level)
        
        logger.add(sys.stdout, format='<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>', colorize=True, level=level)
        logger.info("Using loguru logger with level: "+level)
        return logger
      
    except ImportError:
        level = logging.DEBUG if verbose else logging.INFO
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.info("Using default logger")
        # don't log results for the non-master process
        if distributed_rank > 0:
            return logger
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, filename))
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger
