
import functools
import os
import traceback

from utils import logger


def catch(msg="出现错误，中断训练"):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if os.getenv("DEBUG_MODE") == "true" or os.getenv("DEBUG_MODE") == "1":
                return f(*args, **kwargs)
            try:
                return f(*args, **kwargs)
            except Exception as e:
                logger.info(msg)
                logger.info(str(e))
                _traceback = traceback.format_exc()
                for line in _traceback.split("\n"):
                    logger.info(str(line))
                return None

        return wrapper

    return decorator