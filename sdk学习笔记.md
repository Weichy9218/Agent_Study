# OpenRouter SDK 学习笔记

## 1. 查看模型思考过程 (Reasoning Tokens)

### 什么是 Reasoning Tokens
推理令牌（thinking tokens）提供了对AI模型推理步骤的透明视图，让你看到模型的中间思考过程。

### 支持的模型
- **OpenAI**: o1, o3, GPT-5 系列
- **Anthropic**: Claude 3.7+（使用 `anthropic/claude-3.7-sonnet`）
- **Google**: Gemini thinking 模型
- **其他**: Deepseek, xAI Grok, Qwen 等

### 启用方法

#### 方式1：OpenAI/Grok 模型 - 使用 effort 参数
```python
response = client.chat.completions.create(
    model="openai/o3-mini",
    messages=[{"role": "user", "content": "如何构建世界上最高的摩天大楼？"}],
    extra_body={
        "reasoning": {
            "effort": "high"  # 选项: xhigh, high, medium, low, minimal, none
        }
    }
)

# 查看推理过程
print(response.choices[0].message.reasoning)
```

**Effort 等级说明：**
- `xhigh`: ~95% 的 max_tokens 用于推理
- `high`: ~80%
- `medium`: ~50%
- `low`: ~20%
- `minimal`: ~10%
- `none`: 禁用推理

#### 方式2：Anthropic/Gemini 模型 - 使用 max_tokens 参数
```python
response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=[{"role": "user", "content": "最高效的排序算法是什么？"}],
    extra_body={
        "reasoning": {
            "max_tokens": 2000  # 范围: 1024-32000
        }
    }
)

# 推理内容在 reasoning_details 中
for detail in response.choices[0].message.reasoning_details:
    if detail.type == "reasoning.text":
        print(detail.reasoning)
```

### 隐藏推理过程（仅内部使用）
```python
extra_body={
    "reasoning": {
        "max_tokens": 2000,
        "exclude": True  # 模型使用推理但不返回
    }
}
```

### 多轮对话中保留推理上下文
```python
messages = [
    {"role": "user", "content": "问题1"},
    {
        "role": "assistant",
        "content": "回答1",
        "reasoning_details": previous_reasoning  # 保留之前的推理
    },
    {"role": "user", "content": "问题2"}
]
```

---

## 2. System Prompt 最佳实践

### 官方推荐的 System Prompt 结构

根据 OpenRouter 官方文档和最佳实践：

#### 基础结构
```python
system_prompt = """你是一个专业的助手。

## 你的能力
- 能力1
- 能力2

## 响应格式
请按照以下格式返回：
[具体格式说明]

## 限制
- 限制1
- 限制2
"""
```

#### 通用 System Prompt 模板（基于官方建议）
```python
UNIVERSAL_SYSTEM_PROMPT = """你是一个有帮助、诚实且无害的AI助手。

## 核心原则
1. **准确性**: 提供准确、基于事实的信息
2. **清晰性**: 使用清晰、简洁的语言
3. **完整性**: 提供完整的答案，避免遗漏关键信息
4. **安全性**: 不提供有害、非法或不道德的内容

## 响应指南
- 如果不确定，明确说明你不知道
- 对复杂问题进行分步骤解释
- 在适当时提供示例
- 使用结构化格式（列表、标题）提高可读性

## 特殊情况处理
- 遇到模糊问题时，先澄清用户意图
- 对于编程问题，提供完整可运行的代码
- 对于数据分析，解释你的推理过程
"""
```

### 针对 JSON 输出的 System Prompt
```python
system_prompt = """你必须只返回 JSON 格式的数据。

Schema:
{
    "result": string,
    "confidence": number,  // 0-1
    "reasoning": string
}

重要:
1. 不要在 JSON 之外添加任何文字
2. 确保 JSON 格式正确
3. 所有字符串使用双引号
"""
```

### Prompt Caching 优化策略

**关键原则：** 将不变的内容放在前面，变化的内容放在后面

```python
messages = [
    {
        "role": "system",
        "content": LONG_SYSTEM_PROMPT  # 大量静态内容，会被缓存
    },
    {
        "role": "user",
        "content": RAG_CONTEXT  # 大量检索到的上下文，会被缓存
    },
    {
        "role": "user",
        "content": user_question  # 变化的用户问题，放在最后
    }
]
```

**缓存建议：**
- 缓存大型内容：CSV文件、角色卡片、RAG数据、书籍章节
- Anthropic Claude: 默认5分钟缓存（ephemeral），可设置1小时
- Google Gemini: 保持消息数组初始部分一致以获得缓存命中

---

## 3. 工具调用 (Tool Calling)

### 工作流程
1. **定义工具** → 2. **模型建议调用** → 3. **你执行工具** → 4. **返回结果给模型** → 5. **模型生成最终答案**

### 完整示例

```python
from openai import OpenAI
import json
import requests

client = OpenAI(
    api_key="your-openrouter-api-key",
    base_url="https://openrouter.ai/api/v1"
)

# 1. 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": "在古登堡计划中搜索书籍。当用户想要查找特定书籍时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "搜索关键词列表"
                    }
                },
                "required": ["search_terms"]
            }
        }
    }
]

# 2. 实现工具函数
def search_books(search_terms):
    """实际执行搜索的函数"""
    url = "https://gutendex.com/books"
    response = requests.get(url, params={"search": " ".join(search_terms)})

    results = []
    for book in response.json().get("results", [])[:5]:
        results.append({
            "id": book.get("id"),
            "title": book.get("title"),
            "authors": [a.get("name") for a in book.get("authors", [])]
        })
    return results

# 3. 发起请求
messages = [
    {"role": "user", "content": "帮我找一些关于人工智能的书"}
]

response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # auto | none | {"type": "function", "function": {"name": "tool_name"}}
)

# 4. 处理工具调用
message = response.choices[0].message
if message.tool_calls:
    messages.append(message)  # 添加助手的回复

    for tool_call in message.tool_calls:
        # 解析参数
        args = json.loads(tool_call.function.arguments)

        # 执行工具
        if tool_call.function.name == "search_books":
            result = search_books(args["search_terms"])

        # 添加工具结果
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False)
        })

    # 5. 获取最终答案
    final_response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=messages,
        tools=tools  # 必须在每次请求中包含！
    )

    print(final_response.choices[0].message.content)
```

### 工具调用最佳实践

1. **清晰的函数名**: 使用描述性名称表明用途
2. **详细的描述**: 说明何时以及如何使用该工具
3. **完整的参数模式**: 包含类型、描述和示例
4. **错误处理**: 优雅处理工具执行错误
5. **每次请求都包含 tools**: 路由器需要验证工具模式

### 并行工具调用控制
```python
response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=messages,
    tools=tools,
    parallel_tool_calls=False  # True: 并行执行, False: 串行执行
)
```

### 交错思考 (Interleaved Thinking)
支持的模型可以在工具调用之间进行推理，实现更复杂的决策链：

```
用户问题 → [思考] → 调用工具1 → [思考] → 调用工具2 → [思考] → 最终答案
```

---

## 4. 流式输出 (Streaming)

### 基础流式输出

```python
response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=[{"role": "user", "content": "写一个关于AI的故事"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### 流式输出 + 获取使用统计

```python
response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
    stream_options={"include_usage": True}
)

for chunk in response:
    # 处理内容
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

    # 最后一个chunk包含使用统计
    if chunk.usage:
        print(f"\n\n使用的tokens: {chunk.usage.total_tokens}")
```

### 流式输出中的工具调用

```python
response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=messages,
    tools=tools,
    stream=True
)

tool_calls = []
for chunk in response:
    delta = chunk.choices[0].delta

    # 处理工具调用
    if delta.tool_calls:
        for tc in delta.tool_calls:
            # 收集工具调用信息
            if tc.index >= len(tool_calls):
                tool_calls.append({
                    "id": tc.id,
                    "function": {"name": tc.function.name, "arguments": ""}
                })

            if tc.function.arguments:
                tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

    # 检查结束原因
    if chunk.choices[0].finish_reason == "tool_calls":
        # 处理收集到的工具调用
        for tool_call in tool_calls:
            args = json.loads(tool_call["function"]["arguments"])
            # 执行工具...
```

### 取消流式请求

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-key",
    base_url="https://openrouter.ai/api/v1",
    timeout=30.0
)

# 使用 timeout 自动取消
try:
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=[{"role": "user", "content": "长文本生成..."}],
        stream=True
    )

    for chunk in response:
        print(chunk.choices[0].delta.content, end="")
except Exception as e:
    print(f"请求被取消: {e}")
```

### 流式错误处理

```python
for chunk in response:
    # 检查finish_reason
    if chunk.choices[0].finish_reason == "error":
        error_data = chunk.choices[0].delta
        print(f"错误: {error_data}")
        break

    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## 5. Context 管理

### 多轮对话管理

```python
class ConversationManager:
    def __init__(self, system_prompt, max_history=10):
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.messages = [{"role": "system", "content": system_prompt}]

    def add_user_message(self, content):
        self.messages.append({"role": "user", "content": content})
        self._trim_history()

    def add_assistant_message(self, content):
        self.messages.append({"role": "assistant", "content": content})
        self._trim_history()

    def _trim_history(self):
        # 保留 system prompt + 最近的 N 条消息
        if len(self.messages) > self.max_history + 1:
            self.messages = [self.messages[0]] + self.messages[-(self.max_history):]

    def get_messages(self):
        return self.messages

# 使用
manager = ConversationManager(
    system_prompt="你是一个有帮助的助手",
    max_history=20
)

manager.add_user_message("你好")
response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=manager.get_messages()
)
manager.add_assistant_message(response.choices[0].message.content)
```

### 超长上下文处理 - Middle-Out 压缩

当提示超过最大上下文长度时，OpenRouter 支持自动压缩：

```python
response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=very_long_messages,
    extra_body={
        "transforms": ["middle-out"]  # 自动压缩中间部分
    }
)
```

### Token 计数与成本追踪

```python
# 查询生成信息
import requests

generation_id = response.id  # 从响应中获取

url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
headers = {"Authorization": f"Bearer {api_key}"}

gen_info = requests.get(url, headers=headers).json()

print(f"总tokens: {gen_info['tokens_prompt'] + gen_info['tokens_completion']}")
print(f"成本: ${gen_info['total_cost']}")
```

---

## 6. 高级参数控制

### 温度和采样参数

```python
response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=messages,
    temperature=0.7,      # 0.0-2.0, 控制随机性
    top_p=0.9,           # 0.0-1.0, 核采样
    top_k=40,            # 限制候选tokens
    frequency_penalty=0.0,  # -2.0 to 2.0, 减少重复
    presence_penalty=0.0,   # -2.0 to 2.0, 鼓励新话题
    max_tokens=1000      # 最大生成长度
)
```

### Structured Outputs (JSON 模式)

```python
from pydantic import BaseModel

class BookRecommendation(BaseModel):
    title: str
    author: str
    reason: str
    rating: float

response = client.chat.completions.create(
    model="anthropic/claude-3.7-sonnet",
    messages=[
        {"role": "system", "content": "你是图书推荐专家"},
        {"role": "user", "content": "推荐一本科幻小说"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "book_recommendation",
            "schema": BookRecommendation.model_json_schema()
        }
    }
)

# 解析结构化输出
recommendation = BookRecommendation.model_validate_json(
    response.choices[0].message.content
)
```

---

## 核心要点总结

### 1. 查看思考过程
- 使用 `reasoning` 参数（effort 或 max_tokens）
- Claude 3.7+、OpenAI o-系列、Gemini thinking 模型支持
- 推理内容在 `reasoning_details` 中

### 2. System Prompt
- 遵循：能力说明 + 响应格式 + 限制条件
- 静态内容放前面以利用缓存
- JSON 输出需要明确的 schema 说明

### 3. 工具调用
- 三步流程：定义 → 建议 → 执行 → 返回 → 最终答案
- 每次请求必须包含 `tools` 参数
- 支持并行和串行执行

### 4. 流式输出
- `stream=True` 启用
- `stream_options={"include_usage": True}` 获取统计
- 检查 `finish_reason` 处理错误

### 5. Context 管理
- 限制历史消息数量
- 使用 `transforms: ["middle-out"]` 压缩超长上下文
- 通过 `/api/v1/generation` 追踪 token 使用

---

## 参考资源

- [OpenRouter 官方文档](https://openrouter.ai/docs)
- [Reasoning Tokens 指南](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Tool Calling 文档](https://openrouter.ai/docs/guides/features/tool-calling)
- [Streaming API 参考](https://openrouter.ai/docs/api/reference/streaming)
- [Prompt Caching 最佳实践](https://openrouter.ai/docs/guides/best-practices/prompt-caching)
- [OpenRouter GitHub 示例](https://github.com/OpenRouterTeam/openrouter-examples-python)
