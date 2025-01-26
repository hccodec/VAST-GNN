import logging, re, os


class RemoveANSIColorFilter(logging.Filter):
    ansi_escape = re.compile(f"\x1B\[[0-?]*[ -/]*[@-~]")

    def filter(self, record):
        record.msg = self.ansi_escape.sub("", record.msg)
        return True


# 格式
class ExpFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, end=""):
        super().__init__(fmt, datefmt)
        self.end = end

    def format(self, record):
        record.name = f"{record.name:>15s}"
        return super().format(record)


# 使用单例模式维护 logger
class SingletonLogger:
    _instance = None

    def __new__(cls, log_file_path=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls.logger = None
            cls._instance.set_logger(log_file_path)
        return cls._instance

    def set_logger(self, log_file_path):
        self.logger = logging.getLogger("exp_logger")
        self.file_logger = logging.getLogger("exp_file_logger")
        self.console_logger = logging.getLogger("exp_console_logger")

        fmt_str_console = f"%(message)s \033[2m[%(asctime)s] " + (f"[{os.path.dirname(log_file_path)}]" if log_file_path is not None else '') + "\033[0m"
        fmt_str_file = "[%(asctime)s %(name)s %(levelname)s] %(message)s"
        fmt_datestr_console = "%Y-%m-%d %H:%M:%S"

        self.logger.setLevel(logging.DEBUG)
        self.file_logger.setLevel(logging.DEBUG)
        self.console_logger.setLevel(logging.DEBUG)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(fmt_str_console, fmt_datestr_console)
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        self.console_logger.addHandler(console_handler)

        if log_file_path is None:
            return
        # 处理文件日志
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = ExpFormatter(fmt_str_file)
        file_handler.setFormatter(file_format)
        file_handler.addFilter(RemoveANSIColorFilter())
        self.logger.addHandler(file_handler)
        self.file_logger.addHandler(file_handler)

    def getLogger(self, type=""):
        assert type in ["", "console", "file"], "请选择正确的 logger type"
        if type == "":
            return self.logger
        elif type == "console":
            return self.console_logger
        elif type == "file":
            return self.file_logger


_logger = SingletonLogger()


def set_logger(log_file_path):
    if not os.path.exists(log_file_path):
        os.makedirs(log_file_path, exist_ok=True)
    _logger.set_logger(os.path.join(log_file_path, "log.txt"))


logger = _logger.getLogger()
file_logger = _logger.getLogger("file")
console_logger = _logger.getLogger("console")
