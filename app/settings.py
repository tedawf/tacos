from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def choose_env_file() -> str:
    return ".env.local" if Path(".env.local").exists() else ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=choose_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # tolerate unrelated env vars
    )

    # CouchDB
    COUCHDB_HOST: str = "localhost"
    COUCHDB_PORT: int = 5984
    COUCHDB_USERNAME: str = "admin"
    COUCHDB_PASSWORD: str = ""
    COUCHDB_DATABASE: str = "obsidian_db"

    # Postgres
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USERNAME: str = "postgres"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_DATABASE: str = "postgres_db"

    # Blog
    BLOG_PREFIX: str = "blog/"
    BLOG_API_URL: str = "http://localhost:8000"
    BASE_BLOG_URL: str = "http://localhost:3000"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Our own API Key
    TACOS_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    # RAG
    RAG_CHAT_MODEL: str = "gpt-5-mini"

    # Knowledge Base
    KB_PREFIX: str = "kb/"

    # Portfolio
    PORTFOLIO_PREFIX: str = "portfolio/"

    @property
    def couchdb_url(self) -> str:
        return f"http://{self.COUCHDB_USERNAME}:{self.COUCHDB_PASSWORD}@{self.COUCHDB_HOST}:{self.COUCHDB_PORT}"

    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USERNAME}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DATABASE}"


# Global settings instance (evaluated at import, but reads env on construction)
settings = Settings()
