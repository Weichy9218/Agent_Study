"""
LLM Client for One-API
基于 OpenAI 标准的 LLM 客户端，支持通过 One-API 转发调用
支持结构化输出、上下文缓存、工具调用等高级功能
"""

import os
from typing import Optional, List, Dict, Any, Generator, Type
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv


class LLMClient:
    """LLM 客户端，支持同步和流式调用"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-3.5-turbo"
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥，如果不提供则从环境变量 OPENROUTER_API_KEY 读取
            base_url: API 基础地址，如果不提供则从环境变量 OPENROUTER_BASE_URL 读取
            model: 使用的模型名称，默认为 gpt-3.5-turbo
        """
        # 加载环境变量
        load_dotenv()

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL")
        self.model = model

        if not self.api_key:
            raise ValueError("API key is required. Set OPENROUTER_API_KEY in .env or pass api_key parameter")

        if not self.base_url:
            raise ValueError("Base URL is required. Set OPENROUTER_BASE_URL in .env or pass base_url parameter")

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        发送聊天请求（非流式）

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称，如果不提供则使用初始化时的模型
            temperature: 温度参数，控制随机性 (0-2)
            max_tokens: 最大返回 token 数
            **kwargs: 其他 OpenAI API 支持的参数

        Returns:
            模型返回的文本内容
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            **kwargs
        )

        return response.choices[0].message.content

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        发送聊天请求（流式）

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            model: 模型名称，如果不提供则使用初始化时的模型
            temperature: 温度参数，控制随机性 (0-2)
            max_tokens: 最大返回 token 数
            **kwargs: 其他 OpenAI API 支持的参数

        Yields:
            模型返回的文本片段
        """
        stream = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def simple_chat(self, user_message: str, system_message: Optional[str] = None, **kwargs) -> str:
        """
        简化的聊天接口

        Args:
            user_message: 用户消息
            system_message: 系统消息（可选）
            **kwargs: 其他参数传递给 chat 方法

        Returns:
            模型返回的文本内容
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        return self.chat(messages, **kwargs)

    def list_models(self, grouped: bool = True) -> List[str]:
        """
        列出所有可用模型

        Args:
            grouped: 是否按公司分组排序

        Returns:
            模型名称列表
        """
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]

            if grouped:
                def company_key(model_id: str):
                    if "/" in model_id:
                        company = model_id.split("/", 1)[0]
                        return (0, company.lower(), model_id.lower())
                    return (1, "", model_id.lower())

                return sorted(model_ids, key=company_key)
            else:
                return sorted(model_ids)
        except Exception as e:
            print(f"Failed to fetch models: {e}")
            return []

    def get_available_models(self) -> List[str]:
        """
        获取可用的模型列表（别名方法）

        Returns:
            模型名称列表
        """
        return self.list_models()

    def structured_output(
        self,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        model: Optional[str] = None,
        **kwargs
    ) -> BaseModel:
        """
        使用结构化输出（Structured Output）

        Args:
            messages: 消息列表
            response_format: Pydantic 模型，定义期望的输出结构
            model: 模型名称，建议使用 gpt-4o-mini, gpt-4o 等支持结构化输出的模型
            **kwargs: 其他参数

        Returns:
            返回符合 Pydantic 模型结构的对象

        Example:
            class Answer(BaseModel):
                answer: int
                explanation: str

            result = client.structured_output(
                messages=[{"role": "user", "content": "1+1等于几？"}],
                response_format=Answer
            )
            print(result.answer)  # 2
        """
        response = self.client.beta.chat.completions.parse(
            model=model or self.model,
            messages=messages,
            response_format=response_format,
            **kwargs
        )

        return response.choices[0].message.parsed

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        tool_choice: str = "auto",
        **kwargs
    ):
        """
        使用工具调用（Tool Calling / Function Calling）

        Args:
            messages: 消息列表
            tools: 工具定义列表
            model: 模型名称
            tool_choice: 工具选择策略 ("auto", "none", 或指定工具名)
            **kwargs: 其他参数

        Returns:
            完整的响应对象，包含工具调用信息

        Example:
            tools = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市名称"}
                        },
                        "required": ["city"]
                    }
                }
            }]

            response = client.chat_with_tools(
                messages=[{"role": "user", "content": "北京天气怎么样？"}],
                tools=tools
            )
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs
        )

        return response

    def chat_with_cache(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        使用上下文缓存（Prompt Caching）
        注意：需要在消息中使用特殊格式来标记缓存点

        Args:
            messages: 消息列表，可以包含缓存标记
            model: 模型名称
            **kwargs: 其他参数

        Returns:
            模型返回的文本内容

        Example:
            # 在长文本内容中使用缓存
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "你是一个助手"},
                        {"type": "text", "text": "这里是大量的上下文...", "cache_control": {"type": "ephemeral"}}
                    ]
                },
                {"role": "user", "content": "问题"}
            ]

            注意：cache_control 只在 Claude 模型中支持，OpenAI 模型会自动缓存
        """
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            **kwargs
        )

        return response.choices[0].message.content

    def chat_with_search(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        带搜索功能的聊天（仅支持特定模型）

        注意：搜索功能由模型本身决定，需要使用支持搜索的模型
        例如：gpt-4o-with-search, deepseek-chat 等

        Args:
            user_message: 用户消息
            system_message: 系统消息
            model: 模型名称（必须使用支持搜索的模型）
            **kwargs: 其他参数

        Returns:
            模型返回的文本内容

        Example:
            # 使用支持搜索的模型
            response = client.chat_with_search(
                "TCL科技今天的股价是多少？",
                model="gpt-4o-with-search"
            )
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        return self.chat(messages, model=model, **kwargs)
