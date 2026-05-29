"""
测试智谱API调用
"""
from openai import OpenAI

# 你的智谱API Key
api_key = "5847ee7498034321bd83d2cb293ac3a3.5sR9kyadjZhGr9Hh"
base_url = "https://open.bigmodel.cn/api/paas/v4"

print(f"API Key: {api_key[:20]}...")
print(f"Base URL: {base_url}")

try:
    client = OpenAI(api_key=api_key, base_url=base_url)
    print("Client created successfully")

    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[{"role": "user", "content": "你好，请简短回复"}],
        max_tokens=50,
        timeout=30
    )

    print("Success!")
    print(f"Response: {response.choices[0].message.content}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
