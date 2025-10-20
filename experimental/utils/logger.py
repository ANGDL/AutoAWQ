import logging
import sys

__all__ = ['awq_logger']

class Logger:
    _instance = None
    LEVEL_MAP = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger('AutoAWQ')
            self.logger.setLevel(logging.INFO)
            self.logger.propagate = False  # Prevent double logging
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def enable_file_logging(self, file_path):
        # Remove existing file handlers
        for handler in self.logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                self.logger.removeHandler(handler)
        
        file_handler = logging.FileHandler(file_path)
        formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def set_level(self, level):
        if isinstance(level, str):
            level = self.LEVEL_MAP.get(level.upper(), logging.DEBUG)
        self.logger.setLevel(level)

    def set_name(self, name):
        self.logger.name = name

    def debug(self, msg):
        self.logger.debug(msg, stacklevel=2)

    def info(self, msg):
        self.logger.info(msg, stacklevel=2)

    def warning(self, msg):
        self.logger.warning(msg, stacklevel=2)

    def error(self, msg):
        self.logger.error(msg, stacklevel=2)

    def critical(self, msg):
        self.logger.critical(msg, stacklevel=2)

awq_logger = Logger()
