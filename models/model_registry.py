from typing import Dict, Any
from models.base_model import BaseModel
from models.groq_model import GroqModel
from models.local_model import LocalModel
from models.compatibility import ChatModelCompatibilityError

class ModelRegistry:
    @staticmethod
    def get_model(model_config: Dict[str, Any], global_config: Dict[str, Any] = None) -> BaseModel:
        name = model_config.get("name")
        model_type = model_config.get("type", "groq")
        
        if model_type == "groq":
            return GroqModel(name, global_config)
        elif model_type == "local":
            return LocalModel(name, global_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
