from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

import os
import typing


class Config(BaseModel):
    xai_api_key: str = Field()
    xai_model_name: str = Field()

    tavily_api_key: str = Field()

    max_llm_retries: int = Field(default=3)
    max_researcher_iterations: int = Field(default=3)
    max_concurrent_research_units: int = Field(default=3)

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig | None = None) -> "Config":
        keys = cls.model_fields.keys()

        cfg: dict[str, typing.Any] = {
            key: (
                config.get(key, os.environ.get(key.upper()))
                if config
                else os.environ.get(key.upper())
            )
            for key in keys
        }

        return cls(**cfg)
