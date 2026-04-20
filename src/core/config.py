from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    mongo_uri: str = "mongodb://localhost:27017"
    openai_api_key: SecretStr | None = None
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
