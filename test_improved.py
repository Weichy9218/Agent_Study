"""
改进版 test.py - 应用 OpenRouter 最佳实践
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")

if not api_key:
    raise ValueError("Missing API key. Set OPENROUTER_API_KEY (or OPENAI_API_KEY) in .env.")

client = OpenAI(api_key=api_key, base_url=base_url)

# 用户提示
USER_PROMPT = r"""
You are an agent that can predict future events. The event to be predicted: "2025-12-27, what will be the "most recent" number for "Waiting for response" of this date in the Support Stats published by Steam?\n        IMPORTANT: Your final answer MUST end with this exact format:\n        \\boxed{YOUR_PREDICTION}\n        Do not use any other format. Do not refuse to make a prediction. Do not say \"I cannot predict the future.\" You must make a clear prediction based on the best data currently available, using the box format specified above.
"""

# 定义输出结构
class Prediction(BaseModel):
    prediction_value: float = Field(..., description="预测的数值")
    reasoning_short: str = Field(..., description="简短的推理过程")
    confidence: float = Field(..., ge=0.0, le=1.0, description="预测的置信度 0-1")


# ============================================================================
# 方法 1: 原始方法（你的 test.py）
# ============================================================================
def method_1_original():
    """原始方法 - 基本的JSON输出"""
    print("=" * 60)
    print("方法 1: 原始方法")
    print("=" * 60)

    resp = client.chat.completions.create(
        model="openai/gpt-5.2",
        messages=[
            {
                "role": "system",
                "content": (
                    "Return JSON only. Schema: "
                    '{"prediction_value": number, "reasoning_short": string, "confidence": number}'
                ),
            },
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    content = resp.choices[0].message.content or ""
    print("\n【模型输出】:")
    print(content)

    try:
        pred = Prediction.model_validate_json(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        pred = Prediction.model_validate(json.loads(content[start : end + 1]))

    print(f"\n【结果】: \\boxed{{{pred.prediction_value:.2f}}}")
    return pred


# ============================================================================
# 方法 2: 使用 Reasoning Tokens 查看思考过程
# ============================================================================
def method_2_with_reasoning():
    """改进方法 - 显示模型的思考过程"""
    print("\n" + "=" * 60)
    print("方法 2: 使用 Reasoning Tokens 查看思考过程")
    print("=" * 60)

    # 使用 Claude 3.7 的 reasoning 能力
    resp = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=[
            {
                "role": "system",
                "content": (
                    "Return JSON only. Schema: "
                    '{"prediction_value": number, "reasoning_short": string, "confidence": number}\n\n'
                    "Important: Provide your detailed reasoning first, then output the final JSON."
                ),
            },
            {"role": "user", "content": USER_PROMPT},
        ],
        extra_body={
            "reasoning": {
                "max_tokens": 2000  # 分配tokens给思考过程
            }
        }
    )

    message = resp.choices[0].message

    # 显示思考过程
    if hasattr(message, 'reasoning_details') and message.reasoning_details:
        print("\n【模型的详细思考过程】:")
        for detail in message.reasoning_details:
            if detail.type == "reasoning.text":
                print(detail.reasoning)
                print()

    content = message.content or ""
    print("【最终JSON输出】:")
    print(content)

    # 解析结果
    try:
        pred = Prediction.model_validate_json(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        pred = Prediction.model_validate(json.loads(content[start : end + 1]))

    print(f"\n【结果】: \\boxed{{{pred.prediction_value:.2f}}}")
    print(f"【置信度】: {pred.confidence:.2%}")
    return pred


# ============================================================================
# 方法 3: 使用优化的 System Prompt + Structured Outputs
# ============================================================================
def method_3_structured_output():
    """改进方法 - 使用结构化输出和优化的提示"""
    print("\n" + "=" * 60)
    print("方法 3: 优化的 System Prompt + Structured Outputs")
    print("=" * 60)

    # 优化的 system prompt
    system_prompt = """你是一个专业的数据预测专家。

## 你的能力
- 分析历史数据和趋势
- 基于数据做出合理的预测
- 评估预测的置信度

## 输出格式
你必须返回严格的 JSON 格式，不要添加任何其他文字。

Schema:
{
    "prediction_value": number,      // 预测的数值
    "reasoning_short": string,       // 推理过程（1-2句话）
    "confidence": number            // 置信度 0.0-1.0
}

## 重要规则
1. 基于可用数据做出最佳预测
2. 在 reasoning_short 中简要说明预测依据
3. confidence 应反映预测的不确定性
4. 确保输出的是有效的 JSON 格式
"""

    resp = client.chat.completions.create(
        model="openai/gpt-5.2",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT},
        ],
        response_format={"type": "json_object"},  # 强制 JSON 输出
        temperature=0.3,  # 降低温度以获得更一致的输出
    )

    content = resp.choices[0].message.content or ""
    print("\n【模型输出】:")
    print(content)

    # 解析结果
    pred = Prediction.model_validate_json(content)

    print(f"\n【预测值】: {pred.prediction_value:.2f}")
    print(f"【推理】: {pred.reasoning_short}")
    print(f"【置信度】: {pred.confidence:.2%}")
    print(f"\n【结果】: \\boxed{{{pred.prediction_value:.2f}}}")
    return pred


# ============================================================================
# 方法 4: 使用工具调用获取实时数据
# ============================================================================
def method_4_with_tools():
    """改进方法 - 使用工具调用获取实际数据"""
    print("\n" + "=" * 60)
    print("方法 4: 使用工具调用获取历史数据")
    print("=" * 60)

    # 定义工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_historical_stats",
                "description": "获取Steam支持统计的历史数据。返回过去几天的'等待响应'数量。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "days": {
                            "type": "integer",
                            "description": "要获取的历史天数"
                        }
                    },
                    "required": ["days"]
                }
            }
        }
    ]

    # 模拟工具函数
    def get_historical_stats(days: int):
        """模拟获取历史数据"""
        # 实际应用中这里会调用真实的API
        import random
        base_value = 1500
        stats = []
        for i in range(days):
            value = base_value + random.randint(-200, 200)
            stats.append({
                "date": f"2025-12-{27-days+i}",
                "waiting_for_response": value
            })
            base_value = value
        return {"historical_data": stats}

    # 初始请求
    messages = [
        {
            "role": "system",
            "content": """你是数据预测专家。当需要历史数据时，使用提供的工具获取。

基于历史数据进行预测后，返回 JSON 格式：
{"prediction_value": number, "reasoning_short": string, "confidence": number}
"""
        },
        {"role": "user", "content": USER_PROMPT},
    ]

    print("\n【第一步：请求数据】")
    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message

    # 处理工具调用
    if message.tool_calls:
        print("【模型决定调用工具获取历史数据】")
        messages.append(message)

        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具: {function_name}")
            print(f"参数: {json.dumps(function_args, ensure_ascii=False)}")

            # 执行工具
            if function_name == "get_historical_stats":
                result = get_historical_stats(**function_args)
                print(f"获取到 {len(result['historical_data'])} 天的历史数据")

            # 添加工具结果
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })

        # 获取最终预测
        print("\n【第二步：基于数据进行预测】")
        final_response = client.chat.completions.create(
            model="anthropic/claude-3.7-sonnet",
            messages=messages,
            tools=tools,
            response_format={"type": "json_object"}
        )

        content = final_response.choices[0].message.content or ""
        print("\n【预测结果】:")
        print(content)

        pred = Prediction.model_validate_json(content)
        print(f"\n【预测值】: {pred.prediction_value:.2f}")
        print(f"【推理】: {pred.reasoning_short}")
        print(f"【置信度】: {pred.confidence:.2%}")
        print(f"\n【结果】: \\boxed{{{pred.prediction_value:.2f}}}")
        return pred


# ============================================================================
# 方法 5: 流式输出（实时显示思考过程）
# ============================================================================
def method_5_streaming():
    """改进方法 - 流式输出，实时看到模型的思考"""
    print("\n" + "=" * 60)
    print("方法 5: 流式输出（实时显示）")
    print("=" * 60)

    system_prompt = """你是预测专家。请先详细说明你的分析过程，然后在最后输出 JSON 格式的预测。

JSON Schema: {"prediction_value": number, "reasoning_short": string, "confidence": number}
"""

    response = client.chat.completions.create(
        model="anthropic/claude-3.7-sonnet",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT},
        ],
        stream=True,
        stream_options={"include_usage": True}
    )

    print("\n【实时输出】:")
    full_content = ""

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            full_content += content

        # 显示token使用情况
        if chunk.usage:
            print(f"\n\n【Token 统计】:")
            print(f"输入: {chunk.usage.prompt_tokens}")
            print(f"输出: {chunk.usage.completion_tokens}")
            print(f"总计: {chunk.usage.total_tokens}")

    # 解析JSON
    try:
        start = full_content.find("{")
        end = full_content.rfind("}")
        if start != -1 and end != -1:
            json_str = full_content[start:end+1]
            pred = Prediction.model_validate_json(json_str)
            print(f"\n【结果】: \\boxed{{{pred.prediction_value:.2f}}}")
            return pred
    except Exception as e:
        print(f"\n解析错误: {e}")


# ============================================================================
# 主函数 - 运行所有方法
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" OpenRouter SDK 最佳实践对比演示")
    print("=" * 70)

    methods = [
        ("原始方法", method_1_original),
        ("显示思考过程", method_2_with_reasoning),
        ("优化提示词", method_3_structured_output),
        ("工具调用", method_4_with_tools),
        ("流式输出", method_5_streaming),
    ]

    results = {}

    for name, method in methods:
        try:
            print(f"\n正在运行: {name}")
            result = method()
            results[name] = result
            print(f"\n✓ {name} 完成")
            input("\n按回车键继续下一个示例...")
        except Exception as e:
            print(f"\n✗ {name} 出错: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    print("\n" + "=" * 70)
    print(" 总结")
    print("=" * 70)
    print("\n各方法的优势:")
    print("\n1. 原始方法: 简单直接，适合基础使用")
    print("2. 显示思考过程: 透明度高，可以看到模型的推理步骤")
    print("3. 优化提示词: 输出更稳定，格式更规范")
    print("4. 工具调用: 可以获取实时数据，预测更准确")
    print("5. 流式输出: 用户体验好，实时看到生成过程")
    print("\n建议:")
    print("- 生产环境: 使用方法3（优化提示词）或方法4（工具调用）")
    print("- 调试阶段: 使用方法2（显示思考）或方法5（流式输出）")
    print("- 简单任务: 使用方法1（原始方法）")
    print("=" * 70)
