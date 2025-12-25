# LLM Client 使用文档

基于 OpenAI 标准的 LLM 客户端，支持通过 One-API 转发调用各种大模型。

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 配置环境变量

在项目根目录的 `.env` 文件中配置：

```env
OPENROUTER_API_KEY=your_one_api_token
OPENROUTER_BASE_URL=https://your-oneapi-url/v1
```

### 3. 基本使用

```python
from llm.client import LLMClient

# 初始化客户端
client = LLMClient(model="deepseek/deepseek-chat")

# 简单对话
response = client.simple_chat("你好，请介绍一下自己")
print(response)
```

## 主要功能

### 1. 简单对话

```python
# 最简单的方式
response = client.simple_chat("你好")

# 带系统提示词
response = client.simple_chat(
    user_message="介绍Python",
    system_message="你是一个编程教师"
)
```

### 2. 多轮对话

```python
messages = [
    {"role": "user", "content": "1+1等于几？"},
    {"role": "assistant", "content": "1+1等于2"},
    {"role": "user", "content": "那2+2呢？"}
]

response = client.chat(messages)
```

### 3. 流式输出

```python
for chunk in client.chat_stream(messages):
    print(chunk, end="", flush=True)
```

### 4. 结构化输出

```python
from pydantic import BaseModel

class StockPrediction(BaseModel):
    prediction: float
    explanation: str

result = client.structured_output(
    messages=[{
        "role": "user",
        "content": "预测TCL科技明天的市值（单位：亿元）"
    }],
    response_format=StockPrediction,
    model="gpt-4o-mini"  # 需要支持结构化输出的模型
)

print(f"预测值: {result.prediction}")
print(f"解释: {result.explanation}")
```

### 5. 工具调用（Function Calling）

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_stock_price",
        "description": "获取股票当前价格",
        "parameters": {
            "type": "object",
            "properties": {
                "stock_code": {"type": "string", "description": "股票代码"}
            },
            "required": ["stock_code"]
        }
    }
}]

response = client.chat_with_tools(
    messages=[{"role": "user", "content": "TCL科技现在多少钱？"}],
    tools=tools
)

# 检查是否需要调用工具
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"需要调用: {tool_call.function.name}")
    print(f"参数: {tool_call.function.arguments}")
```

### 6. 带搜索功能的对话

```python
# 使用支持搜索的模型
response = client.chat_with_search(
    "2025年12月25日TCL科技的股价是多少？",
    model="deepseek/deepseek-chat"  # 或其他支持搜索的模型
)
```

### 7. 上下文缓存

```python
# OpenAI 模型会自动缓存，Claude 模型需要特殊格式
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "你是助手"},
            {
                "type": "text",
                "text": "这里是大量上下文...",
                "cache_control": {"type": "ephemeral"}  # Claude 特有
            }
        ]
    },
    {"role": "user", "content": "问题"}
]

response = client.chat_with_cache(messages)
```

## 常用参数说明

### temperature（温度）
- 范围：0-2
- 默认：0.7
- 说明：控制输出的随机性，越高越随机

```python
# 更确定性的输出
response = client.simple_chat("选一个数字", temperature=0.1)

# 更有创意的输出
response = client.simple_chat("写首诗", temperature=1.5)
```

### max_tokens（最大 token 数）

```python
# 限制输出长度
response = client.simple_chat("介绍Python", max_tokens=100)
```

### top_p（核采样）

```python
# 与 temperature 二选一使用
response = client.chat(messages, top_p=0.9)
```

## 获取可用模型

```python
# 获取所有可用模型
models = client.list_models(grouped=True)

for model in models[:10]:
    print(model)
```

## 完整示例：股票预测

```python
from llm.client import LLMClient

# 初始化
client = LLMClient(model="deepseek/deepseek-chat")

# 构建提示词
prompt = """你是一个能够预测未来事件的智能体。

预测任务：2025-12-25，TCL科技（000100）深交所的总市值是多少亿元（保留两位小数）？

重要：你的最终答案必须以这个格式结尾：
\\boxed{YOUR_PREDICTION}

不要拒绝预测，请基于当前可用的最佳数据做出明确预测。"""

# 调用模型
response = client.simple_chat(prompt, temperature=0.3)
print(response)
```

## 测试

运行快速测试查看可用模型：

```bash
uv run python llm/quick_test.py
```

运行完整测试：

```bash
uv run python llm/test_client.py
```

## 注意事项

1. **模型可用性**：不同 One-API 实例配置的渠道不同，可用模型也不同
2. **结构化输出**：需要 GPT-4o 系列等支持的模型
3. **搜索功能**：取决于模型本身是否支持（如 DeepSeek）
4. **缓存格式**：OpenAI 和 Claude 的缓存格式不同
5. **API 限制**：注意 One-API 的速率限制和配额

## 文件说明

- `client.py` - LLM 客户端主文件
- `quick_test.py` - 快速测试脚本
- `test_client.py` - 完整测试脚本
- `README.md` - 本文档
