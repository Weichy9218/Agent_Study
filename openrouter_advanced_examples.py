"""
OpenRouter SDK 高级功能示例
基于 test.py 的完整示例代码
"""

import json
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# 初始化客户端
api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")

if not api_key:
    raise ValueError("Missing API key. Set OPENROUTER_API_KEY in .env.")

client = OpenAI(api_key=api_key, base_url=base_url)


# ============================================================================
# 示例 1: 查看模型思考过程 (Reasoning Tokens)
# ============================================================================

def example_1_reasoning_tokens():
    """演示如何查看Claude和OpenAI模型的中间思考过程"""
    print("=" * 60)
    print("示例 1: 查看模型思考过程")
    print("=" * 60)

    # 方式1: Claude 3.7 使用 max_tokens
    print("\n方式1: Claude 3.7 Sonnet (使用 reasoning.max_tokens)")
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=[
            {
                "role": "user",
                "content": "如何设计一个高效的分布式缓存系统？请详细思考。"
            }
        ],
        extra_body={
            "reasoning": {
                "max_tokens": 2000  # 分配2000个tokens给思考过程
            }
        }
    )

    message = response.choices[0].message

    # 打印思考过程
    if hasattr(message, 'reasoning_details') and message.reasoning_details:
        print("\n【模型的思考过程】:")
        for detail in message.reasoning_details:
            if detail.type == "reasoning.text":
                print(detail.reasoning)

    # 打印最终回答
    print("\n【最终回答】:")
    print(message.content)

    # 方式2: OpenAI o3-mini 使用 effort
    print("\n\n方式2: OpenAI o3-mini (使用 reasoning.effort)")
    try:
        response = client.chat.completions.create(
            model="openai/o3-mini",
            messages=[
                {
                    "role": "user",
                    "content": "解释快速排序算法的时间复杂度"
                }
            ],
            extra_body={
                "reasoning": {
                    "effort": "high"  # high effort = 80% tokens 用于思考
                }
            }
        )

        message = response.choices[0].message
        if hasattr(message, 'reasoning') and message.reasoning:
            print("\n【推理过程】:")
            print(message.reasoning)

        print("\n【最终回答】:")
        print(message.content)
    except Exception as e:
        print(f"注意: {e}")
        print("(某些模型可能需要特定的API权限)")


# ============================================================================
# 示例 2: 优化的 System Prompt
# ============================================================================

# 通用 System Prompt 模板（基于官方最佳实践）
UNIVERSAL_SYSTEM_PROMPT = """你是一个专业、准确、有帮助的AI助手。

## 核心原则
1. **准确性**: 提供准确、基于事实的信息
2. **清晰性**: 使用清晰、简洁的语言
3. **完整性**: 提供完整的答案，避免遗漏关键信息
4. **诚实性**: 如果不确定，明确说明你不知道

## 响应指南
- 对复杂问题进行分步骤解释
- 在适当时提供具体示例
- 使用结构化格式（列表、标题）提高可读性
- 对技术问题提供完整可运行的代码

## 特殊情况处理
- 遇到模糊问题时，先澄清用户意图
- 对于编程问题，考虑边缘情况和错误处理
- 对于数据分析，解释你的推理过程
"""

# 针对 JSON 输出的优化 System Prompt
JSON_SYSTEM_PROMPT_TEMPLATE = """你必须只返回有效的 JSON 格式数据，不要添加任何其他文字。

Schema:
{schema}

重要规则:
1. 严格遵循上述 JSON schema
2. 不要在 JSON 之外添加任何解释或文字
3. 确保 JSON 格式完全正确
4. 所有字符串使用双引号
5. 数字不要使用引号
"""


def example_2_optimized_system_prompt():
    """演示优化的 System Prompt 用法"""
    print("\n" + "=" * 60)
    print("示例 2: 优化的 System Prompt")
    print("=" * 60)

    # 定义输出结构
    class PredictionResult(BaseModel):
        prediction: float = Field(..., description="预测的数值")
        reasoning: str = Field(..., description="推理过程")
        confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0-1")

    schema = PredictionResult.model_json_schema()

    # 使用优化的 system prompt
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=[
            {
                "role": "system",
                "content": JSON_SYSTEM_PROMPT_TEMPLATE.format(
                    schema=json.dumps(schema, indent=2, ensure_ascii=False)
                )
            },
            {
                "role": "user",
                "content": "预测明天北京的最高气温（摄氏度）"
            }
        ]
    )

    content = response.choices[0].message.content
    print("\n【模型输出】:")
    print(content)

    # 解析为结构化数据
    try:
        result = PredictionResult.model_validate_json(content)
        print("\n【解析结果】:")
        print(f"预测值: {result.prediction}°C")
        print(f"置信度: {result.confidence}")
        print(f"推理: {result.reasoning}")
    except Exception as e:
        print(f"解析错误: {e}")


# ============================================================================
# 示例 3: 工具调用 (Tool Calling)
# ============================================================================

def example_3_tool_calling():
    """演示完整的工具调用流程"""
    print("\n" + "=" * 60)
    print("示例 3: 工具调用")
    print("=" * 60)

    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息。当用户询问天气时使用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，例如：北京、上海"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "温度单位"
                        }
                    },
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "执行数学计算。支持基本的算术运算。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "数学表达式，例如：2 + 3 * 4"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    # 实现工具函数
    def get_weather(city: str, unit: str = "celsius") -> Dict[str, Any]:
        """模拟天气API"""
        # 实际应用中这里会调用真实的天气API
        weather_data = {
            "北京": {"temp": 15, "condition": "晴朗"},
            "上海": {"temp": 20, "condition": "多云"},
            "深圳": {"temp": 25, "condition": "阴天"}
        }
        data = weather_data.get(city, {"temp": 18, "condition": "未知"})

        if unit == "fahrenheit":
            data["temp"] = data["temp"] * 9/5 + 32

        return {
            "city": city,
            "temperature": data["temp"],
            "condition": data["condition"],
            "unit": unit
        }

    def calculate(expression: str) -> Dict[str, Any]:
        """安全的计算器"""
        try:
            # 注意：eval 有安全风险，生产环境应使用安全的表达式解析器
            result = eval(expression, {"__builtins__": {}}, {})
            return {"expression": expression, "result": result}
        except Exception as e:
            return {"expression": expression, "error": str(e)}

    # 工具映射
    available_functions = {
        "get_weather": get_weather,
        "calculate": calculate
    }

    # 初始对话
    messages = [
        {
            "role": "system",
            "content": UNIVERSAL_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "北京现在天气怎么样？如果气温是15度，转换成华氏度是多少？"
        }
    ]

    print("\n【用户问题】:", messages[-1]["content"])

    # 第一次请求
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # 处理工具调用
    if message.tool_calls:
        print("\n【模型建议调用工具】:")
        messages.append(message)  # 添加助手的消息

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"- 工具: {function_name}")
            print(f"  参数: {json.dumps(function_args, ensure_ascii=False)}")

            # 执行工具
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            print(f"  结果: {json.dumps(function_response, ensure_ascii=False)}")

            # 添加工具结果到对话
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(function_response, ensure_ascii=False)
            })

        # 获取最终答案
        final_response = client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=messages,
            tools=tools  # 必须在每次请求中包含
        )

        print("\n【最终回答】:")
        print(final_response.choices[0].message.content)
    else:
        print("\n【直接回答】:")
        print(message.content)


# ============================================================================
# 示例 4: 流式输出
# ============================================================================

def example_4_streaming():
    """演示流式输出"""
    print("\n" + "=" * 60)
    print("示例 4: 流式输出")
    print("=" * 60)

    print("\n【流式生成故事】:")
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=[
            {
                "role": "user",
                "content": "用50字写一个关于程序员的小故事"
            }
        ],
        stream=True,
        stream_options={"include_usage": True}
    )

    full_content = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_content += content

        # 打印使用统计（最后一个chunk）
        if chunk.usage:
            print(f"\n\n【Token 使用统计】:")
            print(f"输入: {chunk.usage.prompt_tokens}")
            print(f"输出: {chunk.usage.completion_tokens}")
            print(f"总计: {chunk.usage.total_tokens}")


# ============================================================================
# 示例 5: 流式输出 + 工具调用
# ============================================================================

def example_5_streaming_with_tools():
    """演示流式输出中的工具调用"""
    print("\n" + "=" * 60)
    print("示例 5: 流式输出 + 工具调用")
    print("=" * 60)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "搜索产品数据库",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    messages = [
        {"role": "user", "content": "帮我找一下关于Python的产品"}
    ]

    print("\n【流式响应】:")
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=messages,
        tools=tools,
        stream=True
    )

    tool_calls = []
    content = ""

    for chunk in response:
        delta = chunk.choices[0].delta

        # 处理文本内容
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content

        # 处理工具调用
        if delta.tool_calls:
            for tc in delta.tool_calls:
                # 初始化工具调用
                if tc.index >= len(tool_calls):
                    tool_calls.append({
                        "id": tc.id,
                        "function": {"name": tc.function.name, "arguments": ""}
                    })

                # 累积参数
                if tc.function.arguments:
                    tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

        # 检查结束原因
        finish_reason = chunk.choices[0].finish_reason
        if finish_reason == "tool_calls":
            print("\n\n【检测到工具调用】:")
            for tool_call in tool_calls:
                print(f"工具: {tool_call['function']['name']}")
                print(f"参数: {tool_call['function']['arguments']}")

        elif finish_reason == "stop":
            print("\n\n【响应完成】")


# ============================================================================
# 示例 6: Context 管理
# ============================================================================

class ConversationManager:
    """对话上下文管理器"""

    def __init__(self, system_prompt: str, max_history: int = 20):
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def _trim_history(self):
        """保持对话历史在限制内"""
        # 保留 system prompt + 最近的 N 条消息
        if len(self.messages) > self.max_history + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]

    def get_messages(self) -> List[Dict[str, str]]:
        return self.messages

    def get_token_estimate(self) -> int:
        """粗略估算token数量（每个字符约0.5个token）"""
        total_chars = sum(len(msg["content"]) for msg in self.messages)
        return int(total_chars * 0.5)


def example_6_context_management():
    """演示对话上下文管理"""
    print("\n" + "=" * 60)
    print("示例 6: Context 管理")
    print("=" * 60)

    # 创建对话管理器
    manager = ConversationManager(
        system_prompt=UNIVERSAL_SYSTEM_PROMPT,
        max_history=10
    )

    # 模拟多轮对话
    conversations = [
        "你好，我想学习Python",
        "什么是变量？",
        "如何定义函数？",
        "我之前问过什么问题？"  # 测试上下文记忆
    ]

    for user_input in conversations:
        print(f"\n【用户】: {user_input}")

        manager.add_user_message(user_input)

        response = client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=manager.get_messages(),
            max_tokens=200
        )

        assistant_message = response.choices[0].message.content
        manager.add_assistant_message(assistant_message)

        print(f"【助手】: {assistant_message}")

        # 显示上下文统计
        token_estimate = manager.get_token_estimate()
        print(f"【上下文】: {len(manager.messages)-1} 条消息, 约 {token_estimate} tokens")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OpenRouter SDK 高级功能示例")
    print("=" * 60)

    # 运行所有示例
    try:
        example_1_reasoning_tokens()
    except Exception as e:
        print(f"\n示例 1 错误: {e}")

    try:
        example_2_optimized_system_prompt()
    except Exception as e:
        print(f"\n示例 2 错误: {e}")

    try:
        example_3_tool_calling()
    except Exception as e:
        print(f"\n示例 3 错误: {e}")

    try:
        example_4_streaming()
    except Exception as e:
        print(f"\n示例 4 错误: {e}")

    try:
        example_5_streaming_with_tools()
    except Exception as e:
        print(f"\n示例 5 错误: {e}")

    try:
        example_6_context_management()
    except Exception as e:
        print(f"\n示例 6 错误: {e}")

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("详细学习笔记请查看: sdk学习笔记.md")
    print("=" * 60)
