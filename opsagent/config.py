from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    environment: str = "development"
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://opsagent:opsagent@localhost:5432/opsagent"

    # AI / LLM
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Redis (Phase 2)
    redis_url: str = "redis://localhost:6379/0"

    # Slack
    slack_bot_token: str = ""
    slack_channel: str = ""
    slack_signing_secret: str = ""      # Required for Slack interactive bot (Phase 2)

    # RAG
    runbooks_path: str = "/app/runbooks"

    # Phase 3: action catalog
    actions_path: str = "/app/actions"

    # Phase 4: MLflow experiment tracking
    mlflow_tracking_uri: str = ""

    # Phase 6: JWT auth + Stripe billing
    jwt_secret: str = ""                # Set JWT_SECRET in production
    stripe_secret_key: str = ""         # sk_live_... or sk_test_...
    stripe_webhook_secret: str = ""     # whsec_...
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_retrieval_k: int = 3

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
