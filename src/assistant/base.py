from langchain_core.language_models import BaseChatModel

from .llm import LLM_REGISTRY


def register_llm(model_type):
    def decorator(cls):
        # 检查是否继承自 Assistant
        if not issubclass(cls, BaseChatModel):
            raise TypeError(
                f"Class {cls.__name__} must inherit from langchain_core.language_models.BaseChatModel "
                f"to be registered as {model_type}"
            )
        LLM_REGISTRY[model_type] = cls
        return cls
    return decorator

def get_chat_model(cfg: dict):
    if 'base_url' in cfg and 'model_type' not in cfg:
        if cfg['base_url'].strip().startswith('http'):
            model_type = 'oai'
            cfg['model_type'] = model_type
            return LLM_REGISTRY[model_type](cfg)

    if cfg['model_type'] in LLM_REGISTRY:
        return LLM_REGISTRY[cfg['model_type']](cfg)

    raise ValueError(f'Invalid model cfg: {cfg}')
