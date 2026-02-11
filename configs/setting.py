from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Dataset(BaseModel):
    kaggle_username: str = ""
    kaggle_key: SecretStr = SecretStr("")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_nested_delimiter="_",
        env_nested_max_split=1,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    dataset: Dataset = Dataset()


@lru_cache
def get_settings() -> Settings:
    return Settings()
