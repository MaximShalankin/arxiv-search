"""Structured JSON logging.

Writes one JSON object per line to stderr (always) and to a file (optional).
Fields: ts, level, stage, msg, plus any kwargs.

Usage:
    log = get_logger(stage="parser")
    log.info("parsed", arxiv_id="2306.12345", latency_s=4.9, markdown_chars=12453)
"""
import json
import logging
import sys
import time
from pathlib import Path


class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "stage": getattr(record, "stage", "-"),
            "msg": record.getMessage(),
        }
        # Attach extra fields (everything in __dict__ that's not standard)
        std = {"name", "msg", "args", "levelname", "levelno", "pathname", "filename",
               "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
               "created", "msecs", "relativeCreated", "thread", "threadName",
               "processName", "process", "stage", "taskName"}
        for k, v in record.__dict__.items():
            if k not in std:
                try:
                    json.dumps(v)  # serializable check
                    payload[k] = v
                except (TypeError, ValueError):
                    payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(stage: str = "-", level: str = "INFO", log_file: Path | None = None):
    """Return a logger that emits JSON lines tagged with `stage`.

    Idempotent: calling twice with same name returns the same logger,
    re-attaching handlers only if missing.
    """
    logger = logging.getLogger(f"arxiv_mvp.{stage}")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(JsonFormatter())
        logger.addHandler(fh)

    # Adapter: add stage to every record via extra, and merge arbitrary
    # kwargs (e.g. log.info("x", foo=1, bar=2)) into extra so JsonFormatter
    # can serialize them.
    _STANDARD_KWARGS = {"exc_info", "stack_info", "extra", "msg", "args"}

    # Reserved LogRecord attributes — if user passes these as kwargs, logging
    # will raise KeyError("Attempt to overwrite ... in LogRecord"). Rename
    # collisions to f"{key}_" so the data survives.
    _RESERVED = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "taskName",
    })

    class _StageAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            extra = kwargs.setdefault("extra", {})
            extra["stage"] = self.logger.name.replace("arxiv_mvp.", "")
            # Move arbitrary user kwargs into extra, renaming reserved-name collisions
            for key in list(kwargs.keys()):
                if key in _STANDARD_KWARGS:
                    continue
                safe_key = f"{key}_" if key in _RESERVED else key
                extra[safe_key] = kwargs.pop(key)
            return msg, kwargs

    return _StageAdapter(logger, {})
