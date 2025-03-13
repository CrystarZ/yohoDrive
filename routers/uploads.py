import os
from fastapi import APIRouter, HTTPException, UploadFile, Form
from pydantic import FilePath
from . import pwd
from db.mysql.database import database as mysql
from db.mysql.models import Upload, User
from .users import c_fd_user, find_user
from config import decoder as conf
from utils.files import SrcType, uniqueFileName, checkFileType

conf_db = conf(f"{pwd}/config.conf").Section("database").dict
conf_be = conf(f"{pwd}/config.conf").Section("backend").dict
router = APIRouter()


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
        fn = savename
        with open(savepath, "wb") as buffer:
            buffer.write(file.file.read())
    return fn


@router.post("/files/upload")
async def upload_file(file: UploadFile, user_id: int = Form()):
    db = mysql(**conf_db)
    try:
        user = c_fd_user(id=user_id)
        u = find_user(user)
        if not u:
            raise HTTPException(status_code=404, detail="User not found")
        user_id = u.id
        fn = save_upload_file(file)
        if fn is not None:
            up = Upload(filename=file.filename, filepath=fn, user_id=user_id)
            db.add(up)
            db.refresh(up)
            return up
        else:
            raise HTTPException(status_code=500)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
