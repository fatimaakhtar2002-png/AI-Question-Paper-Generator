import os
from huggingface_hub import InferenceClient

token = os.environ.get("HF_TOKEN")

if not token:
    print("ERROR: HF_TOKEN nahi mila.")
    exit()

client = InferenceClient(
    api_key=token,
    provider="auto"
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "system",
            "content": "You are an educational question paper generator."
        },
        {
            "role": "user",
            "content": """
Generate 5 questions about Python Programming.
Difficulty: Medium.

Give:
- 2 MCQs with 4 options each
- 2 Short Answer questions
- 1 Long Answer question
"""
        }
    ],
    max_tokens=1000
)

print("\nAI GENERATED QUESTIONS:\n")
print(response.choices[0].message.content)