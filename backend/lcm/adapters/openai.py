from .base import BackendAdapter


class OpenAICompatibleAdapter(BackendAdapter):
    kind = "openai"
    label = "OpenAI-compatible"


class LMStudioAdapter(OpenAICompatibleAdapter):
    kind = "lmstudio"
    label = "LM Studio"


class LlamaCppAdapter(OpenAICompatibleAdapter):
    kind = "llamacpp"
    label = "llama.cpp"


class VLLMAdapter(OpenAICompatibleAdapter):
    kind = "vllm"
    label = "vLLM"


class MLXAdapter(OpenAICompatibleAdapter):
    kind = "mlx"
    label = "MLX / MLX-LM"


class XTLLMAdapter(OpenAICompatibleAdapter):
    kind = "xtllm"
    label = "XTLLM / xLLM"
