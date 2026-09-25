import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime

from langsmith import get_current_run_tree

request_id: ContextVar[str] = ContextVar("request_id", default="-")
logger = logging.getLogger("zania")


def configure_logging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    # SDK debug output may contain uploaded data. Application logs use an explicit field list.
    for name in ("openai", "httpx", "httpcore", "pypdf"):
        logging.getLogger(name).setLevel(logging.CRITICAL)


def event(stage: str, **fields: object) -> None:
    logger.info(json.dumps({"request_id": request_id.get(), "stage": stage, **fields}))
    run = get_current_run_tree()
    if run is not None:
        run.add_event({"name": stage, "time": datetime.now(UTC).isoformat(), "kwargs": fields})
