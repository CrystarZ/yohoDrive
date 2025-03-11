from typing import Optional, Dict
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base


class database:
    def __init__(
        self,
        username: str = "root",
        password: str = "password",
        host: str = "127.0.0.1",
        port: int = 3306,
        dbname: str = "test",
    ) -> None:
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.dbname = dbname

        self.engine = create_engine(self.URL, echo=True)
        self.session = self.Session()

    @property
    def Session(self):
        """
        生成新会话
        """
        return sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

    @property
    def URL(self) -> str:
        """
        生成数据库连接 URL。
        """
        return f"mysql+pymysql://{self.username}:{self.password}@{self.host}:{self.port}/{self.dbname}"

    def add(self, instance):
        self.session.add(instance)
        self.session.commit()
        return instance

    def query(self, _entity):
        return self.session.query(_entity)

    def create_tables(self):
        Base.metadata.create_all(self.engine)

    def recreate_tables(self):
        Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
