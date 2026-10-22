from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    xai_api_key: str = ""
    llm_provider: str = "openrouter"
    llm_model: str = "nvidia/nemotron-3-super-120b-a12b:free"
    api_key: str = "dev-key"
    port: int = 8000
    max_retries: int = 3
    max_tool_calls_per_task: int = 15
    workspace_dir: str = ""

    # --- Safety / reliability (added with the eval harness) ---
    permission_mode: str = "ask"          # off | ask | enforce
    tool_max_retries: int = 2
    tool_retry_base_delay: float = 0.5
    tool_circuit_breaker_threshold: int = 3
    tool_circuit_breaker_cooldown: float = 30.0
    tool_output_max_chars: int = 8000
    max_context_tokens: int = 24000

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
