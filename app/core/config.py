"""
Application configuration using Pydantic Settings.
Loads configuration from environment variables.
"""
from typing import Optional, Any, List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    APP_NAME: str = "GenoLens Next"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", description="Environment: development, staging, production")
    DEBUG: bool = Field(default=True, description="Debug mode")

    # API
    API_V1_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/genolens",
        description="PostgreSQL connection URL"
    )

    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection string"
    )

    # Celery
    CELERY_BROKER_URL: Optional[str] = Field(
        default=None,
        description="Celery broker URL (defaults to REDIS_URL)"
    )
    CELERY_RESULT_BACKEND: Optional[str] = Field(
        default=None,
        description="Celery result backend (defaults to REDIS_URL)"
    )

    # Supabase
    SUPABASE_URL: str = Field(..., description="Supabase project URL")
    SUPABASE_KEY: str = Field(..., description="Supabase anon key")
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = Field(None, description="Supabase service role key (for admin ops)")
    SUPABASE_JWT_SECRET: str = Field(..., description="Supabase JWT secret for token validation")
    SUPABASE_STORAGE_BUCKET: str = Field(
        default="genolens-data",
        description="Supabase storage bucket name"
    )

    # CORS
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins"
    )

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.strip().startswith("["):
            return [i.strip() for i in v.split(",")]
        return v
    
    # File Processing
    LOCAL_STORAGE_PATH: str = Field(
        default="/app/data",
        description="Local path for storing dataset files"
    )
    MAX_UPLOAD_SIZE: int = Field(
        default=500 * 1024 * 1024,  # 500 MB
        description="Maximum file upload size in bytes"
    )
    ALLOWED_FILE_EXTENSIONS: list[str] = Field(
        default=[".csv", ".tsv", ".xlsx", ".txt"],
        description="Allowed file extensions for upload"
    )

    # Data Processing
    PARQUET_COMPRESSION: str = Field(
        default="snappy",
        description="Parquet compression algorithm"
    )
    CHUNK_SIZE: int = Field(
        default=10000,
        description="Chunk size for processing large files"
    )

    @property
    def celery_broker(self) -> str:
        """Get Celery broker URL, defaulting to Redis URL."""
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_backend(self) -> str:
        """Get Celery result backend URL, defaulting to Redis URL."""
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"


# Global settings instance
settings = Settings()
