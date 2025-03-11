from sqlalchemy import Column, ForeignKey, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime, timezone

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    email = Column(String(100), unique=True, nullable=True)
    updated_at = Column(DateTime, default=datetime.now(timezone.utc))
    registried_at = Column(DateTime, default=datetime.now(timezone.utc))
    avatar_id = Column(Integer, ForeignKey("uploads.id", use_alter=True))

    def __repr__(self):
        return f"<User(username={self.username}, password={self.password})>"


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, autoincrement=True)
    path = Column(String(255), nullable=False)
    at = Column(DateTime, default=datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey("users.id"))

    def __repe__(self):
        return f"<Upload(path={self.path})>"
