import sys
import os

pwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(pwd)

from db.mysql.database import database as mysql
from db.mysql.models import User
from config import decoder as conf

conf_db = conf(f"{pwd}/config.conf").Section("database").dict

if __name__ == "__main__":
    sql = mysql(**conf_db)
    users = sql.query(User).all()
    for user in users:
        print(user)

    exit(0)
