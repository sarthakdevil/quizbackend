from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def clean_docs(docs_path: Optional[Path] = None) -> int:
    """Delete all files in the `docs` directory (recursively).

    Returns the number of files deleted.
    """
    if docs_path is None:
        docs_path = Path(__file__).resolve().parent / "docs"

    docs_path = Path(docs_path)

    if not docs_path.exists():
        logger.warning("docs directory does not exist: %s", docs_path)
        return 0

    deleted = 0
    # Remove only files (including those in subdirectories). Keep directories.
    for p in docs_path.rglob("*"):
        try:
            if p.is_file():
                p.unlink()
                deleted += 1
                logger.info("Deleted file: %s", p)
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            logger.exception("Failed to delete %s: %s", p, exc)

    logger.info("clean_docs finished: %d files removed from %s", deleted, docs_path)
    return deleted


def start_scheduler(run_at_hour: int = 0, run_at_minute: int = 0) -> BackgroundScheduler:
    """Start a background scheduler that runs clean_docs every day at the given hour/minute.

    By default it runs at 00:00 (midnight) local time.
    """
    scheduler = BackgroundScheduler()

    trigger = CronTrigger(hour=run_at_hour, minute=run_at_minute)
    scheduler.add_job(
        lambda: clean_docs(),
        trigger,
        id="daily_clean_docs",
        replace_existing=True,
        max_instances=1,
    )

    # Start immediately (non-blocking)
    scheduler.start()
    logger.info("Scheduler started: daily clean at %02d:%02d", run_at_hour, run_at_minute)
    return scheduler


def start_interval_scheduler() -> BackgroundScheduler:
    """Start a background scheduler that runs clean_docs every 24 hours from startup.
    
    This runs every 24 hours from when the scheduler is started, not at a specific time.
    """
    scheduler = BackgroundScheduler()

    # Use IntervalTrigger to run every 24 hours (86400 seconds)
    trigger = IntervalTrigger(hours=24)
    scheduler.add_job(
        lambda: clean_docs(),
        trigger,
        id="interval_clean_docs",
        replace_existing=True,
        max_instances=1,
    )

    # Start immediately (non-blocking)
    scheduler.start()
    logger.info("Interval scheduler started: cleanup will run every 24 hours")
    return scheduler


if __name__ == "__main__":
    # When run directly, start scheduler and keep the process alive.
    sched = start_scheduler()
    # Optionally run one immediate cleanup on startup as a convenience.
    logger.info("Running initial cleanup on startup")
    clean_docs()

    try:
        # Sleep loop to keep the background scheduler running in this process.
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler")
        sched.shutdown()
