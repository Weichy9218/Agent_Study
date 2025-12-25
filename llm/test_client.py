"""
LLM Client 测试文件
测试 One-API 转发的 LLM 客户端功能
"""

from client import LLMClient


def test_simple_chat():
    """测试简单对话"""
    print("=" * 50)
    print("测试 1: 简单对话")
    print("=" * 50)

    client = LLMClient()
    response = client.simple_chat("你好，请用一句话介绍你自己")
    print(f"回复: {response}\n")


def test_chat_with_system():
    """测试带系统消息的对话"""
    print("=" * 50)
    print("测试 2: 带系统消息的对话")
    print("=" * 50)

    client = LLMClient()
    response = client.simple_chat(
        user_message="介绍一下 Python",
        system_message="你是一个专业的编程教师，请用简洁易懂的语言回答问题"
    )
    print(f"回复: {response}\n")


def test_multi_turn_chat():
    """测试多轮对话"""
    print("=" * 50)
    print("测试 3: 多轮对话")
    print("=" * 50)

    client = LLMClient()
    messages = [
        {"role": "user", "content": "请告诉我 1+1 等于几"},
        {"role": "assistant", "content": "1+1 等于 2"},
        {"role": "user", "content": "那 2+2 呢？"}
    ]
    response = client.chat(messages)
    print(f"回复: {response}\n")


def test_stream_chat():
    """测试流式对话"""
    print("=" * 50)
    print("测试 4: 流式对话")
    print("=" * 50)

    client = LLMClient()
    messages = [
        {"role": "user", "content": "请写一首关于编程的短诗（4行）"}
    ]

    print("流式回复: ", end="", flush=True)
    for chunk in client.chat_stream(messages):
        print(chunk, end="", flush=True)
    print("\n")


def test_different_models():
    """测试不同模型"""
    print("=" * 50)
    print("测试 5: 获取可用模型列表")
    print("=" * 50)

    client = LLMClient()
    models = client.get_available_models()

    if models:
        print(f"可用模型数量: {len(models)}")
        print(f"前 5 个模型: {models[:5]}")
    else:
        print("未能获取模型列表（可能 API 不支持此功能）")
    print()


def test_with_parameters():
    """测试不同参数"""
    print("=" * 50)
    print("测试 6: 使用不同的温度参数")
    print("=" * 50)

    client = LLMClient()

    print("温度 = 0.1 (更确定性):")
    response1 = client.simple_chat(
        "从 1 到 10 中随机选择一个数字",
        temperature=0.1
    )
    print(f"回复: {response1}\n")

    print("温度 = 1.5 (更随机):")
    response2 = client.simple_chat(
        "从 1 到 10 中随机选择一个数字",
        temperature=1.5
    )
    print(f"回复: {response2}\n")


def main():
    """运行所有测试"""
    print("\n开始测试 LLM Client...\n")

    try:
        test_simple_chat()
        test_chat_with_system()
        test_multi_turn_chat()
        test_stream_chat()
        test_different_models()
        test_with_parameters()

        print("=" * 50)
        print("所有测试完成!")
        print("=" * 50)

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
