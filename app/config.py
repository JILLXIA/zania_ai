from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ZANIA_", env_file=".env", extra="ignore")

    openai_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="OPENAI_API_KEY")
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="LANGSMITH_API_KEY"
    )
    langsmith_project: str = Field(default="zania-demo", validation_alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", validation_alias="LANGSMITH_ENDPOINT"
    )
    langsmith_workspace_id: str = Field(default="", validation_alias="LANGSMITH_WORKSPACE_ID")
    langsmith_hide_inputs: bool = Field(default=True, validation_alias="LANGSMITH_HIDE_INPUTS")
    langsmith_hide_outputs: bool = Field(default=True, validation_alias="LANGSMITH_HIDE_OUTPUTS")
    model_cache: str = ".models"
    max_request_bytes: int = Field(default=21 * 1024 * 1024, gt=0)
    max_document_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_questions_bytes: int = Field(default=256 * 1024, gt=0)
    max_questions: int = Field(default=30, ge=1, le=100)
    max_question_chars: int = Field(default=2000, gt=0)
    max_pages: int = Field(default=200, gt=0)
    max_text_chars: int = Field(default=1_000_000, gt=0)
    max_json_depth: int = Field(default=32, gt=0, le=100)
    max_chunks: int = Field(default=2000, gt=0)
    max_visual_pages: int = Field(default=10, ge=0)
    vision_token_budget: int = Field(default=350_000, gt=0)
    upload_timeout: float = Field(default=60, gt=0)
    parsing_timeout: float = Field(default=20, gt=0)
    rendering_timeout: float = Field(default=20, gt=0)
    vision_timeout: float = Field(default=120, gt=0)
    processing_timeout: float = Field(default=180, gt=0)
    provider_timeout: float = Field(default=30, gt=0)
    max_requests: int = Field(default=2, gt=0)
    max_llm_calls: int = Field(default=6, gt=0)
    calls_per_request: int = Field(default=3, gt=0)


MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
