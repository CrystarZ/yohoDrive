from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, Float
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
    filename = Column(String(255), nullable=False)
    filepath = Column(String(255), nullable=False)
    at = Column(DateTime, default=datetime.now(timezone.utc))
    user_id = Column(Integer, ForeignKey("users.id"))

    def __repr__(self):
        return f"<Upload(name={self.filename},path={self.filepath})>"


class Detections(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    class_name = Column(String(255), nullable=False)
    confidence = Column(Float, nullable=False)
    x_min = Column(Integer, nullable=False)
    y_min = Column(Integer, nullable=False)
    x_max = Column(Integer, nullable=False)
    y_max = Column(Integer, nullable=False)
    tag = Column(String(255), nullable=True)
    num = Column(Integer, nullable=True)
    upload_id = Column(Integer, ForeignKey("uploads.id", use_alter=True))

    def __repr__(self):
        return f"<Class(name={self.class_name},source={self.upload_id})>"


class Locations(Base):
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed = Column(Float, nullable=True)
    at = Column(DateTime, default=datetime.now(timezone.utc))


class UserLog(Base):
    __tablename__ = "userlogs"

    id = Column(Integer, primary_key=True)
    action = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    upload_id = Column(Integer, ForeignKey("uploads.id", use_alter=True))
    location_id = Column(Integer, ForeignKey("locations.id", use_alter=True))
    description = Column(String(255))
    at = Column(DateTime, default=datetime.now(timezone.utc))

    def __repr__(self):
        return f"<UserLog(id={self.id},action={self.action},user={self.user_id})>"
