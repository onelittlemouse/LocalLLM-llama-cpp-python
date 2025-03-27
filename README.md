# LocalLLM-llama-cpp-python
Chat with your local LLM(gguf) in console(pycharm IDE) or html(website "http://localhost:1234/v1/chat-ui"). You need to install anaconda and pycharm to construct the required environment like llama-cpp-python(cpu or cuda), etc.

# Model configuration(Update with your GGUF model path)
MODEL_PATH = "E:/LLM/models/qwen/qwen2_5/qwen2.5-3b-instruct-q8_0.gguf"
CONTEXT_LENGTH = 16384
CPU_THREADS = 24
GPU_LAYERS = -1
VERBOSE = False
MODEL_PARAMS = {
    "n_ctx": CONTEXT_LENGTH,  # Context length
    "n_threads": CPU_THREADS,  # Number of CPU threads to use
    "n_gpu_layers": GPU_LAYERS,  # Number of layers to offload to GPU (if available)
    "verbose": VERBOSE  # Whether to print debug info
}
MAX_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.95
TOP_K = 40
MIN_P = 0.05
STOP = None
ECHO = False
