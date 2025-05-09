import os
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, Form
from pydantic import BaseModel
from . import pwd
from db.mysql.database import database as mysql
from db.mysql.models import Upload, UserLog
from .users import c_fd_user, r_fd_user
from config import decoder as conf
from utils.files import SrcType, uniqueFileName, checkFileType

conf_db = conf(f"{pwd}/config.conf").Section("database").dict
conf_be = conf(f"{pwd}/config.conf").Section("backend").dict
router = APIRouter()


class c_fd_upload(BaseModel):
    id: int | None = None


def fd_up(
    db: mysql,
    id: int | None = None,
) -> Upload | None:
    u = None
    if id is not None:
        u = db.query(Upload).filter_by(id=id).first()
    return u


def r_fd_up(db: mysql, up: c_fd_upload) -> Upload | None:
    return fd_up(db, id=up.id)


def save_dir(t: SrcType, root: str = pwd) -> str:
    d = f"{root}/{conf_be.get('srcpath')}"
    if t is SrcType.default:
        d = f"{d}/default"
    elif t is SrcType.pic:
        d = f"{d}/pics"
    elif t is SrcType.video:
        d = f"{d}/videos"
    elif t is SrcType.avatar:
        d = f"{d}/avatar"
    elif t is SrcType.frame:
        d = f"{d}/frame"
    else:
        return None
    d = os.path.abspath(d)
    os.makedirs(d, exist_ok=True)
    return d


def save_upload_file(file: UploadFile, t: SrcType | None = None) -> str | None:
    fn = file.filename
    if fn is not None:
        if t is None:
            t = checkFileType(fn)
        savename = uniqueFileName(fn)
        savepath = os.path.join(save_dir(t), savename)
        fn = savepath
        with open(savepath, "wb") as buffer:
            buffer.write(file.file.read())
    return fn


def save_frame(filename: str, img: Image.Image):
    fn = filename
    t = SrcType.frame
    savename = uniqueFileName(fn)
    savepath = os.path.join(save_dir(t), savename)
    img.save(savepath)
    return savepath


@router.post("/files/upload")
async def upload_file(file: UploadFile, user_id: int = Form()):
    db = mysql(**conf_db)
    u = r_fd_user(db, c_fd_user(id=user_id))
    if u is None:
        raise HTTPException(status_code=404, detail="未找到指定用户")
    user_id = u.id
    fn = save_upload_file(file)
    if fn is None:
        raise HTTPException(status_code=500)

    try:
        up = Upload(filename=file.filename, filepath=fn, user_id=user_id)
        db.add(up)
        db.refresh(up)

        log = UserLog(user_id=user_id, upload_id=up.id, action="create_upload")
        db.add(log)

        db.refresh(up)
        return up

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.post("/files/upload/fd")
def find_upload(up: c_fd_upload):
    db = mysql(**conf_db)
    u = None
    u = r_fd_up(db, up)
    if u is None:
        raise HTTPException(status_code=404, detail="未找到指定资源")

    try:
        log = UserLog(user_id=u.user_id, upload_id=u.id, action="find_upload")
        db.add(log)

        db.refresh(u)
        return u

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
