import torch

class ResponseGenerator:
    def __init__(self, base_model, assistant_model, tokenizer):
        self.base_model = base_model
        self.assistant_model = assistant_model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:0")

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

        # Move inputs to GPU right after tokenization
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        )
        # Move all tensors in inputs to GPU
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        )
        
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
            # Ensure scores are on GPU
            base_score = base_score.to(self.device)
            assistant_score = assistant_score.to(self.device)
            
            mask = torch.abs(assistant_score) > filter_threshold
            filtered_assistant = assistant_score * mask
            final_score = base_score - alpha * filtered_assistant
            filtered_scores.append(final_score)
        
        # Get the final tokens
        final_tokens = base_outputs.sequences[0]  # Start with base sequence
        
        # Apply the filtered scores to get new tokens
        for i, score in enumerate(filtered_scores):
            next_token = torch.argmax(score, dim=-1)
            if i + 1 < len(final_tokens):
                final_tokens[i + 1] = next_token
        
        # Decode and return
        decoded_text = self.tokenizer.decode(final_tokens, skip_special_tokens=True)
        
        # Remove the prompt from the response if it's included
        if decoded_text.startswith(prompt):
            decoded_text = decoded_text[len(prompt):].strip()
            
        return decoded_text