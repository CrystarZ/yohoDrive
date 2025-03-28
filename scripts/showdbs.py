import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.mysql.database import database as mysql
from db.mysql.models import User, Upload, Detections
from config import decoder as conf

pwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
conf_db = conf(f"{pwd}/config.conf").Section("database").dict


def showusers():
    db = mysql(**conf_db)
    E = db.query(User).all()
    for e in E:
        print(e)
    db.close()


def showuploads():
    db = mysql(**conf_db)
    E = db.query(Upload).all()
    for e in E:
        print(e)
    db.close()


def showdetections():
    db = mysql(**conf_db)
    E = db.query(Detections).all()
    for e in E:
        print(e)
    db.close()


if __name__ == "__main__":
    showusers()
    exit(0)
