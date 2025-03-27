from llama_cpp import Llama


def handle_stream_output(output):
    """
    处理流式输出，将生成的内容逐步打印出来，并收集完整的回复。
    参数：output: 生成器对象，来自 create_chat_completion 的流式输出
    返回：response: 完整的回复文本
    """
    response = ""
    for chunk in output:
        delta = chunk['choices'][0]['delta']
        if 'role' in delta:
            print(f"{delta['role']}: ", end='', flush=True)
        elif 'content' in delta:
            content = delta['content']
            print(content, end='', flush=True)
            response += content
    return response


class ChatSession:
    def __init__(self, llm):
        self.llm = llm
        self.messages = []

    def add_message(self, role, content):
        """
        添加一条消息到会话中。
        参数：role: 消息角色，通常为 'user' 或 'assistant' content: 消息内容
        """
        self.messages.append({"role": role, "content": content})

    def get_response_stream(self, user_input):
        """
        获取模型对用户输入的响应（流式输出）。
        参数：user_input: 用户输入的文本
        返回：response: 完整的回复文本
        """
        self.add_message("user", user_input)
        try:
            output = self.llm.create_chat_completion(
                messages=self.messages,
                stream=True,
                temperature=0.2,
                top_p=0.95,
                top_k=40,
                min_p=0.05,
                typical_p=1.0,
                max_tokens=16384
            )
            response = handle_stream_output(output)  # 同时打印和收集回复
            self.add_message("assistant", response.strip())
            return response.strip()
        except Exception as e:
            print(f"\n发生错误: {e}")
            return ""


# 初始化模型（假设使用本地路径）
# model_path = "E:/LLM/models/meta-llama/meta-llama3_1/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
model_path = "E:/LLM/models/qwen/qwen2_5/qwen2.5-3b-instruct-q8_0.gguf"
llm = Llama(
    model_path=model_path,
    n_gpu_layers=0,  # 根据需要卸载到 GPU(-1 = ALL)
    verbose=False,  # 禁用详细日志输出
    n_ctx=16384
)

# 创建会话实例
chat = ChatSession(llm)


if __name__ == '__main__':
    while True:
        prompt = input("User: ")
        # 退出对话条件（当然，你也可以直接终止代码块）
        if prompt.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        chat.get_response_stream(prompt)
        print()

