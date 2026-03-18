from __future__ import annotations
import dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Optional here so the app can start even if you haven't set it yet.
    # For production, you SHOULD set this in your env file or hosting panel.
    n8n_webhook_url: str | None = None
    allowed_origins: str = "http://localhost:5173"
    log_level: str = "INFO"
    enable_transcribe: bool = False

    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


settings = Settings()

