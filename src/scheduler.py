"""
Daily job runner.

CSE trading hours: 09:30 – 14:30 Sri Lanka time (UTC+5:30).
Jobs run at 15:30 SL time every weekday (Mon–Fri) to ensure the day's
data is fully published before we fetch it.

Run this once and leave it running in the background:
    python -m src.scheduler
"""

import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from . import scraper, predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# CSE closes 14:30 IST = 09:00 UTC.  We fetch at 10:00 UTC (15:30 IST).
FETCH_HOUR_UTC   = 10
FETCH_MINUTE_UTC = 0


def daily_job():
    logger.info("=== Daily CSE data refresh started ===")
    try:
        scraper.refresh_today()
    except Exception as exc:
        logger.error("Scraper failed: %s", exc)

    logger.info("Running predictions …")
    try:
        predictor.predict_all()
    except Exception as exc:
        logger.error("Predictor failed: %s", exc)

    logger.info("=== Daily job complete ===")


def run():
    scheduler = BlockingScheduler(timezone="UTC")
    trigger = CronTrigger(
        day_of_week="mon-fri",
        hour=FETCH_HOUR_UTC,
        minute=FETCH_MINUTE_UTC,
    )
    scheduler.add_job(daily_job, trigger, id="daily_cse_refresh")
    logger.info(
        "Scheduler started — daily job runs Mon–Fri at %02d:%02d UTC (15:30 SL time).",
        FETCH_HOUR_UTC, FETCH_MINUTE_UTC,
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
        sys.exit(0)


if __name__ == "__main__":
    run()
