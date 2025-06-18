from sqlalchemy import Column, Integer, String
from app.db.base_class import Base

class Language(Base):
    __tablename__ = "languages"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    country_code = Column(String, nullable=False)
    country_name = Column(String, nullable=False) 