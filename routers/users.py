from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import Boolean
from . import pwd
from db.mysql.database import database as mysql
from db.mysql.models import User, UserLog
from config import decoder as conf
from utils.isEmail import isEmail

conf_db = conf(f"{pwd}/config.conf").Section("database").dict
router = APIRouter()


class c_reg_user(BaseModel):
    username: str
    password: str
    email: str | None = None


class c_fd_user(BaseModel):
    id: int | None = Query(default=None)
    email: str | None = None
    username: str | None = None


class c_login(BaseModel):
    username: str
    password: str


def fd_user(
    db: mysql,
    id: int | None = None,
    email: str | None = None,
    username: str | None = None,
) -> User | None:
    u = None
    if id is not None:
        u = db.query(User).filter_by(id=id).first()
    elif email is not None:
        u = db.query(User).filter_by(email=email).first()
    elif username is not None:
        u = db.query(User).filter_by(username=username).first()
    return u


def r_fd_user(db: mysql, user: c_fd_user) -> User | None:
    return fd_user(db, id=user.id, username=user.username, email=user.email)


@router.post("/users/regis")
def create_user(user: c_reg_user):
    db = mysql(**conf_db)
    if user.email and not isEmail(user.email):
        print("cant")
        raise HTTPException(status_code=400, detail="邮箱格式不正确")

    u = fd_user(db, username=user.username)
    if u is not None:
        raise HTTPException(status_code=500, detail="用户已存在,指定用户名重复")

    u = fd_user(db, email=user.email)
    if u is not None:
        raise HTTPException(status_code=500, detail="用户已存在,指定邮箱重复")

    try:
        user = User(username=user.username, password=user.password, email=user.email)
        db.add(user)
        db.refresh(user)

        log = UserLog(user_id=user.id, action="regis_user")
        db.add(log)

        db.refresh(user)
        return user

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/users/login")
def login(info: c_login):
    db = mysql(**conf_db)
    name = info.username
    pwd = info.password
    u = fd_user(db, id=None, username=name, email=None)
    if u is None:
        raise HTTPException(status_code=404, detail="未找到指定用户")

    if pwd != u.password:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    try:
        log = UserLog(user_id=u.id, action="login")
        db.add(log)

        db.refresh(u)
        return {"msg": "登录成功", "id": u.id}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/users/bg/fd")  # 后台
def find_user(user: c_fd_user):
    db = mysql(**conf_db)
    u = r_fd_user(db, user)
    if u is None:
        raise HTTPException(status_code=404, detail="未找到指定用户")
    try:
        log = UserLog(user_id=u.id, action="find_user")
        db.add(log)

        db.refresh(u)
        return u

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/users/bg/deregis")
def decreate(user: c_fd_user):
    db = mysql(**conf_db)
    u = r_fd_user(db, user)
    if u is None:
        raise HTTPException(status_code=404, detail="未找到指定用户")

    try:
        db.delete(u)

        log = UserLog(user_id=u.id, action="deregis_user")
        db.add(log)

        db.refresh(u)
        return u
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
