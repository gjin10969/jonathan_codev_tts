from pydantic import BaseModel
from database_config import Base
from sqlalchemy import Boolean, Column, Integer, String, Float


# Pydantic model for input data
# class User(Base):
#     __tablename__ = 'users'
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(50))

class Post(Base):
    __tablename__ = "posts"
    
    id = Column(Integer, primary_key=True, index=True)
    input_audio = Column(String(255))
    model_name = Column(String(50))
    f0_up_key = Column(Integer)
    f0_method = Column(String(100))
    index_rate = Column(Float)
    protect = Column(Float)
    filter_radius = Column(Integer)
    resample_sr = Column(Float)
    rms_mix_rate = Column(Float)
    result_ = Column(String(255))
