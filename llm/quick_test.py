"""
快速测试 - 查找可用模型并测试基本功能
"""

from client import LLMClient

def main():
    print("=" * 60)
    print("LLM Client 快速测试")
    print("=" * 60)

    # 初始化客户端
    client = LLMClient()

    # 1. 列出可用模型
    print("\n[1/3] 获取可用模型...")
    models = client.list_models(grouped=True)

    if not models:
        print("❌ 无法获取模型列表")
        return

    # print(f"✅ 找到 {len(models)} 个模型")
    # for model in models:
    #     print(f"  - {model}")

    # 2. 测试简单对话
    print("\n" + "=" * 60)
    print("[2/3] 测试简单对话...")

    # 尝试几个基础模型
    test_models = [
        "google/gemini-3-flash-preview",
        "openai/gpt-5.2",
        "anthropic/claude-sonnet-4.5",
        "qwen/qwen3-max",
        "moonshotai/kimi-k2",
    ]

    for model in test_models:
        if model not in models and not any(m.endswith(model.split("/")[-1]) for m in models):
            continue

        try:
            print(f"\n尝试模型: {model}")
            client.model = model
            response = client.simple_chat("你好，用一句话介绍自己", max_tokens=100)
            print(f"✅ 成功!")
            print(f"回复: {response}")

        except Exception as e:
            error_msg = str(e)
            if "无可用渠道" in error_msg:
                print(f"❌ 模型不可用")
            else:
                print(f"❌ 错误: {error_msg[:100]}")

    exit()
    # 3. 测试流式输出
    print("\n" + "=" * 60)
    print("[3/3] 测试流式输出...")
    print(f"使用模型: {working_model}\n")

    print("流式回复: ", end="", flush=True)
    try:
        for chunk in client.chat_stream(
            messages=[{"role": "user", "content": "数到5，每个数字用逗号分隔"}],
            max_tokens=50
        ):
            print(chunk, end="", flush=True)
        print("\n✅ 流式输出成功!")
    except Exception as e:
        print(f"\n❌ 流式输出失败: {e}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print(f"建议使用的模型: {working_model}")
    print("=" * 60)

if __name__ == "__main__":
    main()
