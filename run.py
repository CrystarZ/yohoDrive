import os
import logging
from config import decoder as conf

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("yohoDrive")

conf_be = conf("./config.conf").Section("backend").dict
be_run = conf_be["run"]

logger.info(f"{be_run}")
os.system(be_run)
