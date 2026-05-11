import os
import sys
import logging
from rich.logging import RichHandler
from rich.console import Console

# Tạo chung một console
_console = Console()

def get_logger(log_dir=".", log_level="INFO", job_name="default"):
    """
    Create a logger using standard logging + rich.
    """
    logger = logging.getLogger(job_name)
    
    # Chỉ setup nếu logger chưa có handler nào
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Rich Console Handler
        rich_handler = RichHandler(
            console=_console,
            show_time=True,
            omit_repeated_times=False,
            show_level=True,
            show_path=True,
            markup=True
        )
        
        # Định dạng text cho đẹp
        formatter = logging.Formatter("[dim cyan]\\[%(name)s][/dim cyan] %(message)s")
        rich_handler.setFormatter(formatter)
        logger.addHandler(rich_handler)
        
        # Ngăn chặn log bị lặp lại ở root logger
        logger.propagate = False
        
    return logger
