# download_wikitext.py
import requests
import os
import logging
import torch

def download_wikitext(save_dir: str = "data"):
    """Download wikitext-2 test set directly."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Direct link to wikitext-2 test set
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
    
    print("Downloading wikitext-2 dataset...")
    save_path = os.path.join(save_dir, "wikitext2_test.txt")
    
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}")
        return save_path
    
    try:
        import zipfile
        from io import BytesIO
        
        # Download the zip file
        response = requests.get(url)
        with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
            # Extract only the test file
            test_file = [f for f in zip_ref.namelist() if 'test' in f][0]
            zip_ref.extract(test_file, save_dir)
            
            # Rename to our standard name
            os.rename(os.path.join(save_dir, test_file), save_path)
            
        print(f"Successfully downloaded to {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Error downloading wikitext: {str(e)}")
        raise

def evaluate_perplexity(model, tokenizer, logger: logging.Logger) -> float:
    """Calculate perplexity on wikitext2 dataset."""
    data_path = "data/wikitext2_test.txt"
    
    try:
        # Check if file exists
        if not os.path.exists(data_path):
            logger.error(f"Wikitext file not found at {data_path}")
            return float('inf')
            
        # Load text
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        logger.info("Calculating perplexity on wikitext2...")
        
        # Tokenize text
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        
        max_length = 1024  # Adjust based on GPU memory
        nlls = []
        
        for i in range(0, encodings.input_ids.size(1), max_length):
            end_i = min(i + max_length, encodings.input_ids.size(1))
            input_ids = encodings.input_ids[:, i:end_i].to("cuda:0")
            target_ids = input_ids.clone()
            
            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss
                
            nlls.append(neg_log_likelihood)
        
        perplexity = torch.exp(torch.stack(nlls).mean())
        logger.info(f"Perplexity calculation complete: {perplexity:.4f}")
        return perplexity
        
    except Exception as e:
        logger.error(f"Error calculating perplexity: {str(e)}")
        return float('inf')  # Return infinity if calculation fails

if __name__ == "__main__":
    download_wikitext()