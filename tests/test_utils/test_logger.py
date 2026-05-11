import pytest
import logging
from torch_wm.utils.logger import get_logger, get_console, SUCCESS_LEVEL_NUM

def test_get_logger():
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"

def test_logger_success():
    logger = get_logger("test_success")
    # Verify success method added to Logger class
    assert hasattr(logger, "success")
    # Should not crash
    logger.success("Test success message")

def test_get_console():
    console = get_console()
    from rich.console import Console
    assert isinstance(console, Console)

def test_premium_rich_handler_levels():
    from torch_wm.utils.logger import PremiumRichHandler
    handler = PremiumRichHandler()
    
    # Mock log records for different levels
    levels = ["INFO", "WARNING", "ERROR", "CRITICAL", "SUCCESS", "DEBUG"]
    expected_substrings = ["INFO", "WARN", "ERR", "CRIT", "SUCC", "DBG"]
    for level, substr in zip(levels, expected_substrings):
        level_num = getattr(logging, level) if level != "SUCCESS" else SUCCESS_LEVEL_NUM
        record = logging.LogRecord("name", level_num, "pathname", 1, "msg", None, None)
        text = handler.get_level_text(record)
        assert substr in text.plain
