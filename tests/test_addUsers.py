import sys
import os

pwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(pwd)

from db.mysql.database import database as mysql
from db.mysql.models import User
from config import decoder as conf

conf_db = conf(f"{pwd}/config.conf").Section("database").dict
sql = mysql(**conf_db)


def db_add():
    zhang = User(username="zhangsan", password="secure123")
    li = User(username="lisi", password="secure456")
    wang = User(username="wangwu", password="secure789")
    users = [zhang, li, wang]
    for user in users:
        sql.add(user)


if __name__ == "__main__":
    db_add()
