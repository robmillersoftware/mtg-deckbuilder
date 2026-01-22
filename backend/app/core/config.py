from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Spellbook"
    APP_URL: str = "http://localhost:3000"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://spellbook:spellbook@localhost:5432/spellbook"
    DATABASE_URL_SYNC: str = "postgresql://spellbook:spellbook@localhost:5432/spellbook"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # JWT
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None

    # OpenAI (for embeddings)
    OPENAI_API_KEY: Optional[str] = None

    # SendGrid
    SENDGRID_API_KEY: Optional[str] = None
    SENDGRID_FROM_EMAIL: str = "noreply@spellbook.app"

    # Scryfall
    SCRYFALL_BULK_DATA_URL: str = "https://api.scryfall.com/bulk-data"

    # Admin
    ADMIN_EMAIL: str = "admin@spellbook.app"

    # Scheduler
    ENABLE_SCHEDULER: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
