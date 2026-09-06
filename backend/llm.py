# -*- coding: utf-8 -*-
"""DeepSeek 裁决模型适配器。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class DeepSeekConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    timeout: int = 60
    temperature: float = 0.0
    max_retries: int = 2

    @classmethod
    def from_yaml(cls, path: str) -> "DeepSeekConfig":
        with open(path, "r", encoding="utf-8") as file:
            raw = yaml.safe_load(file) or {}
        section = raw.get("deepseek", raw)
        configured = cls(**section)
        api_key = os.getenv("LLM_API_KEY") or configured.api_key
        return cls(**{**configured.__dict__, "api_key": api_key})


class LLMClient:
    """只支持 DeepSeek-V4-Flash 的 LangChain ChatModel 客户端。"""

    def __init__(
        self,
        base_url: str = DeepSeekConfig.base_url,
        api_key: str = "",
        model: str = DeepSeekConfig.model,
        timeout: int = DeepSeekConfig.timeout,
        temperature: float = 0.0,
        max_retries: int = 2,
        config: Optional[DeepSeekConfig] = None,
    ):
        self.config = config or DeepSeekConfig(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            temperature=temperature,
            max_retries=max_retries,
        )
        if self.config.provider != "deepseek" or self.config.model != "deepseek-v4-flash":
            raise ValueError("本项目裁决 LLM 已冻结为 deepseek / deepseek-v4-flash")

    @classmethod
    def from_config(cls, path: str) -> "LLMClient":
        return cls(config=DeepSeekConfig.from_yaml(path))

    def available(self) -> bool:
        return bool(self.config.base_url and self.config.api_key and self.config.model)

    def build_model(self) -> Any:
        """返回 LangChain ChatOpenAI 实例（OpenAI 兼容接口）。"""
        if not self.available():
            raise RuntimeError(
                "DeepSeek 裁决模型未配置：请在 config/llm.yaml 填写 api_key，"
                "或设置环境变量 LLM_API_KEY"
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
        )

