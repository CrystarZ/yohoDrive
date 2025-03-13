from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from . import pwd
from db.mysql.database import database as mysql
from db.mysql.models import User
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


@router.post("/users/regis")
def create_user(user: c_reg_user):
    db = mysql(**conf_db)
    try:
        if user.email and not isEmail(user.email):
            print("cant")
            raise HTTPException(status_code=400, detail="邮箱格式不正确")
        user = User(username=user.username, password=user.password, email=user.email)
        db.add(user)
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/users/bg/fd")  # 后台
def find_user(user: c_fd_user):
    db = mysql(**conf_db)
    u = None
    try:
        if user.id is not None:
            u = db.query(User).filter_by(id=user.id).first()
        elif user.email is not None:
            u = db.query(User).filter_by(email=user.email).first()
        elif user.username is not None:
            u = db.query(User).filter_by(username=user.username).first()
        else:
            print("test")
            raise HTTPException(status_code=400, detail="必须提供 id,username 或 email")
        if u is None:
            raise HTTPException(status_code=404, detail="Not find user")
        return u
    finally:
        db.close()


@router.post("/users/bg/deregis")
def decreate(user: c_fd_user):
    db = mysql(**conf_db)
    try:
        u = find_user(user)
        db.delete(u)
        return u
    except Exception as e:
        raise e
    finally:
        db.close()
