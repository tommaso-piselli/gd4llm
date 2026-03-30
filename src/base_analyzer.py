from pathlib import Path
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from model_selector import ModelSelector
from utils import load_config


class BaseAnalyzer:
    """Base class for all analyzers with common functionality"""

    def __init__(self, settings, config_path: Path, output_folder: Path):
        self.settings = settings
        self.config = load_config(config_path)
        self.output_folder = output_folder
        self.model = self._setup_model()
        self.strategies = {
            "zero_shot": ["standard", "cot"],
            "few_shots": ["standard", "cot"]
        }

    def _setup_model(self):
        """Initialize the model using the model selector"""
        selector = ModelSelector()
        config = self.settings.model.to_config()
        return selector.get_model(config)

    def _load_examples(self, prompt_type: str, reasoning_type: str) -> List[Dict]:
        """Base implementation for loading examples - to be overridden by subclasses"""
        raise NotImplementedError

    def _create_messages(self, *args, **kwargs) -> List:
        """Base implementation for creating messages - to be overridden by subclasses"""
        raise NotImplementedError
