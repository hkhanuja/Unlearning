from .config import ModelConfig, LoRAConfig, TrainingConfig
from .model.base import BaseModel
from .model.assistant import AssistantModel
from .tokenizer import UnlearningTokenizer
from .trainer import UnlearningTrainer
from .generator import ResponseGenerator

class AlpacaUnlearning:
    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        lora_config: LoRAConfig = LoRAConfig(),
        training_config: TrainingConfig = TrainingConfig()
    ):
        self.tokenizer = UnlearningTokenizer(model_config.model_path)
        self.base_model = BaseModel(model_config.model_path, model_config.load_in_8bit)
        self.assistant_model = AssistantModel(
            model_config.model_path,
            model_config.num_assistant_layers,
            model_config.load_in_8bit,
            lora_config
        )
        self.trainer = UnlearningTrainer(self.assistant_model, training_config)
        self.generator = ResponseGenerator(
            self.base_model,
            self.assistant_model,
            self.tokenizer
        )
    
    def train(self, forget_data, checkpoint_dir):
        processed_data = [
            self.tokenizer.prepare_training_batch(item["prompt"], item["response"])
            for item in forget_data
        ]
        self.trainer.train(processed_data, checkpoint_dir=checkpoint_dir)
    
    def generate(self, prompt: str, **kwargs):
        return self.generator.generate(prompt, **kwargs)