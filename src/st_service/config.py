from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    st_model_name: str = Field(default="BAAI/bge-m3", alias="ST_MODEL_NAME")
    st_model_cache_dir: Path | None = Field(default=None, alias="ST_MODEL_CACHE_DIR")
    st_device: str = Field(default="cpu", alias="ST_DEVICE")
    st_host: str = Field(default="0.0.0.0", alias="ST_HOST")
    st_port: int = Field(default=8000, alias="ST_PORT")

    @field_validator("st_model_cache_dir", mode="before")
    @classmethod
    def empty_cache_to_none(cls, v):
        if v is None or v == "":
            return None
        return Path(v)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_torch_device(spec: str):
    import torch

    if spec == "cpu":
        return torch.device("cpu")
    if spec.startswith("cuda"):
        return torch.device(spec)
    raise ValueError(f"Unsupported ST_DEVICE: {spec!r}, use 'cpu' or 'cuda' / 'cuda:N'")
