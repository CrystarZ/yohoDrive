import os
from fastapi import FastAPI
from routers import tests, uploads, users
from config import decoder as conf

pwd = os.path.abspath(os.path.dirname(__file__))
conf_be = conf(f"{pwd}/config.conf").Section("backend").dict
os.makedirs(f"{pwd}/{conf_be['srcpath']}", exist_ok=True)

app = FastAPI()

app.include_router(tests.router)
app.include_router(users.router)
app.include_router(uploads.router)
