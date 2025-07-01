"""
Logging utilities for Claude Cadence with color support
"""

import logging
import sys
from pathlib import Path

# ANSI color codes
class Colors:
    # Reset
    RESET = '\033[0m'

    # Regular colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bold colors
    BOLD = '\033[1m'
    BOLD_RED = '\033[1;31m'
    BOLD_GREEN = '\033[1;32m'
    BOLD_YELLOW = '\033[1;33m'
    BOLD_BLUE = '\033[1;34m'
    BOLD_MAGENTA = '\033[1;35m'
    BOLD_CYAN = '\033[1;36m'
    BOLD_WHITE = '\033[1;37m'

    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages"""

    # Define colors for different log levels
    LEVEL_COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.BOLD_RED
    }

    # Define colors for specific keywords
    KEYWORD_COLORS = {
        'ITERATION': Colors.BOLD_CYAN,
        'SUPERVISOR': Colors.BOLD_BLUE,
        'AGENT': Colors.BOLD_MAGENTA,
        'STARTING': Colors.BOLD_GREEN,
        'COMPLETED': Colors.BOLD_GREEN,
        'ERROR': Colors.BOLD_RED,
        'WARNING': Colors.BOLD_YELLOW,
        'Previous agent result:': Colors.CYAN,
        '===': Colors.BOLD_WHITE,
        '---': Colors.WHITE,
    }

    def __init__(self, *args, use_color=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record):
        # Get the base formatted message
        msg = super().format(record)

        if not self.use_color:
            return msg

        # Apply level color to the level name
        levelname = record.levelname
        if levelname in self.LEVEL_COLORS:
            level_color = self.LEVEL_COLORS[levelname]
            # Color just the level name in the message
            msg = msg.replace(f' - {levelname} - ', f' - {level_color}{levelname}{Colors.RESET} - ')

        # Apply keyword colors
        for keyword, color in self.KEYWORD_COLORS.items():
            if keyword in msg:
                # Special handling for separators
                if keyword in ['===', '---']:
                    # Color entire lines of separators
                    lines = msg.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip() and all(c in '=-' for c in line.strip()):
                            lines[i] = f"{color}{line}{Colors.RESET}"
                    msg = '\n'.join(lines)
                else:
                    # Color the keyword
                    msg = msg.replace(keyword, f"{color}{keyword}{Colors.RESET}")

        return msg


def setup_colored_logging(level=logging.INFO):
    """Set up colored logging for the application"""
    # Remove any existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)

    # Create formatter
    formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add handler to root logger
    root_logger.addHandler(console_handler)
    root_logger.setLevel(level)

    return root_logger


def get_colored_logger(name, level=logging.INFO):
    """Get a logger with colored output"""
    logger = logging.getLogger(name)

    # Only set up if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False

    return logger


def setup_file_logging(session_id: str, component_type: str, log_dir: Path, level=logging.DEBUG):
    """Attach a plain file handler to the component's logger.

    Args:
        session_id: The session ID for organizing logs
        component_type: The component type (orchestrator, supervisor, agent)
        log_dir: The base log directory
        level: The logging level (default: DEBUG)

    Returns:
        The configured logger, or None if setup failed
    """
    try:
        session_dir = log_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        log_file = session_dir / f"{component_type}.log"
        logger = logging.getLogger(component_type)

        # Avoid duplicate handlers when called twice
        if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file) for h in logger.handlers):
            return logger

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
        logger.setLevel(level)
        return logger

    except (OSError, PermissionError) as e:
        # Log to console if file logging fails
        console_logger = logging.getLogger(component_type)
        console_logger.warning(f"Failed to set up file logging for {component_type}: {e}")
        console_logger.warning(f"Continuing with console logging only")
        return None
