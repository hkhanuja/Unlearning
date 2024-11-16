import torch

class ResponseGenerator:
    def __init__(self, base_model, assistant_model, tokenizer):
        self.base_model = base_model
        self.assistant_model = assistant_model
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        alpha: float = 0.75,
        filter_threshold: float = 1e-2,
        max_length: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        self.assistant_model.eval()
        
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.base_model.device)
        
        base_outputs = self.base_model.generate(
            **inputs,
            max_length=max_length,
            output_scores=True,
            return_dict_in_generate=True,
            temperature=temperature,
            **kwargs
        )
        
        assistant_outputs = self.assistant_model.generate(
            **inputs,
            max_length=max_length,
            output_scores=True,
            return_dict_in_generate=True,
            temperature=temperature,
            **kwargs
        )
        
        filtered_scores = []
        for base_score, assistant_score in zip(base_outputs.scores, assistant_outputs.scores):
            mask = torch.abs(assistant_score) > filter_threshold
            filtered_assistant = assistant_score * mask
            final_score = base_score - alpha * filtered_assistant
            filtered_scores.append(final_score)
        
        final_tokens = torch.cat([
            torch.argmax(score, dim=-1) for score in filtered_scores
        ])
        
        return self.tokenizer.decode(final_tokens, skip_special_tokens=True)