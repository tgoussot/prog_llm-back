from pydantic import BaseModel
from typing import Optional

class LanguageBase(BaseModel):
    name: str
    country_code: str
    country_name: str

    class Config:
        from_attributes = True

class LanguageCreate(LanguageBase):
    pass

class Language(LanguageBase):
    id: int

    class Config:
        from_attributes = True 