from pydantic import BaseModel

class MessageResponse(BaseModel):
    """
    Schéma de réponse pour un message simple
    """
    message: str 