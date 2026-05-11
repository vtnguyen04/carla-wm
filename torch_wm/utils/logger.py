import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.text import Text

# 1. Thêm Custom Log Level (SUCCESS)
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kws):
    """Log a success message with premium styling."""
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

logging.Logger.success = success

# 2. Xây dựng Theme chuẩn Production (Màu sắc hài hòa, hiện đại)
custom_theme = Theme({
    "info": "cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "critical": "bold white on red",
    "success": "bold green",
    "debug": "dim white",
    "traceback.border": "red",
})

console = Console(theme=custom_theme)

# 3. Custom Rich Handler để chèn Icon xịn xò
class PremiumRichHandler(RichHandler):
    def get_level_text(self, record: logging.LogRecord) -> Text:
        """Inject modern icons into log levels."""
        level_name = record.levelname
        if level_name == "INFO":
            return Text(" ℹ INFO ", style="black on cyan", justify="center")
        elif level_name == "WARNING":
            return Text(" ⚠ WARN ", style="black on yellow", justify="center")
        elif level_name == "ERROR":
            return Text(" ✖ ERR  ", style="white on red", justify="center")
        elif level_name == "CRITICAL":
            return Text(" ☠ CRIT ", style="bold white on bright_red", justify="center")
        elif level_name == "SUCCESS":
            return Text(" ✔ SUCC ", style="black on green", justify="center")
        elif level_name == "DEBUG":
            return Text(" ⚙ DBG  ", style="black on white", justify="center")
        return super().get_level_text(record)

_BANNER_PRINTED = False

def print_banner():
    """Hiển thị ASCII Art chính thức của dự án (Chỉ in 1 lần)."""
    global _BANNER_PRINTED
    if _BANNER_PRINTED:
        return
    
    banner = (
        "[bold cyan]"
        "   ____           _      __        __         _     _ __  __           _      _ \n"
        "  / ___|__ _ _ __| | __ _\\ \\      / /__  _ __| | __| |  \\/  | ___   __| | ___| |\n"
        " | |   / _` | '__| |/ _` |\\ \\ /\\ / / _ \\| '__| |/ _` | |\\/| |/ _ \\ / _` |/ _ \\ |\n"
        " | |__| (_| | |  | | (_| | \\ V  V / (_) | |  | | (_| | |  | | (_) | (_| |  __/ |\n"
        "  \\____\\__,_|_|  |_|\\__,_|  \\_/\\_/ \\___/|_|  |_|\\__,_|_|  |_|\\___/ \\__,_|\\___|_|\n"
        "[/bold cyan]\n"
        "   [bold green]Transformer World Model for Autonomous Driving[/bold green]\n"
    )
    console.print(banner)
    _BANNER_PRINTED = True

def setup_logger():
    """Cấu hình Root Logger chuẩn Production."""
    import os
    from logging.handlers import RotatingFileHandler

    root_logger = logging.getLogger()
    if any(isinstance(h, PremiumRichHandler) for h in root_logger.handlers):
        return
            
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # 1. Console Handler (Rich)
    rich_handler = PremiumRichHandler(
        console=console,
        show_time=True,
        omit_repeated_times=False,
        show_level=True,
        show_path=True,
        enable_link_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        markup=True
    )
    
    # In tên của logger (Component Name) trước message
    formatter = logging.Formatter("[dim cyan]\\[%(name)s][/dim cyan] %(message)s")
    rich_handler.setFormatter(formatter)
    root_logger.addHandler(rich_handler)
    
    # 2. File Handler (Lưu trữ Log vĩnh viễn)
    os.makedirs("logs", exist_ok=True)
    file_handler = RotatingFileHandler(
        "logs/system.log", maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
    )
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    root_logger.setLevel(logging.INFO)
    
    # In Banner
    print_banner()

setup_logger()

def get_logger(name: str) -> logging.Logger:
    """Trả về Logger đã được cấu hình Premium."""
    return logging.getLogger(name)

def get_console() -> Console:
    """Trả về Console toàn cục để in Bảng/Panel."""
    return console
