from typing import Dict, Any
from models.base_model import BaseModel
from models.huggingface_model import HuggingFaceModel
from models.local_model import LocalModel

class ModelCapabilityError(Exception):
    pass

class ModelRegistry:
    @staticmethod
    def get_model(model_config: Dict[str, Any], global_config: Dict[str, Any] = None) -> BaseModel:
        name = model_config.get("name")
        model_type = model_config.get("type", "huggingface")
        capabilities = model_config.get("capabilities", {"supports_chat": True, "supports_text_generation": True})
        
        if not capabilities.get("supports_chat") and not capabilities.get("supports_text_generation"):
            raise ModelCapabilityError(f"Model {name} must support at least one generation type.")
            
        if model_type == "huggingface":
            return HuggingFaceModel(name, global_config, capabilities)
        elif model_type == "local":
            return LocalModel(name, global_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
