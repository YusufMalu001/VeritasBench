from typing import Dict, Any
from models.base_model import BaseModel
from models.huggingface_model import HuggingFaceModel
from models.local_model import LocalModel

class ModelRegistry:
    @staticmethod
    def get_model(model_config: Dict[str, Any], global_config: Dict[str, Any] = None) -> BaseModel:
        name = model_config.get("name")
        model_type = model_config.get("type", "huggingface")
        
        if model_type == "huggingface":
            return HuggingFaceModel(name, global_config)
        elif model_type == "local":
            return LocalModel(name, global_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
