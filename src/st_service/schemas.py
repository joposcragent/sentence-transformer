from pydantic import BaseModel


class TextCorpus(BaseModel):
    text: str


class VectorsPair(BaseModel):
    left: list[float]
    right: list[float]


class CosineSimilarityResponse(BaseModel):
    similarity: float
