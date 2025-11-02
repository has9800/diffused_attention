import torch

# This is what GPT-5 Codex will run
def eval_language_modeling(model, dataset, context_lengths):
    results = []
    for context_len in context_lengths:
        total_loss = 0
        num_batches = 0
        
        for batch in dataset:
            input_ids = batch['input_ids'][:, :context_len]
            
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        results.append({
            'context_length': context_len,
            'perplexity': perplexity.item(),
        })
    
    return results

def eval_passkey_retrieval(model, context_lengths=[32768, 65536]):
    results = []
    
    for context_len in context_lengths:
        correct = 0
        total = 10  # 10 examples per length
        
        for trial in range(total):
            # Generate synthetic passkey task
            target_key = f"XXXXX{trial}YYYYY"
            random_text_len = context_len - len(target_key) - 50
            
            # Create prompt
            prompt = "[Random text] ... " + target_key + " ... [More random text]"
            prompt = prompt[:context_len]
            
            # Ask model to extract
            response = model.generate(prompt, max_new_tokens=20)
            
            if target_key in response:
                correct += 1
        
        accuracy = correct / total
        results.append({
            'context_length': context_len,
            'accuracy': accuracy,
        })
    
    return results

from nltk.translate.bleu_score import sentence_bleu

def eval_code_completion(model, dataset):
    results = []
    
    for example in dataset:
        code = example['code']
        context = code[:len(code)//2]  # First half as context
        target = code[len(code)//2:]   # Second half to predict
        
        # Generate prediction
        prediction = model.generate(context, max_new_tokens=len(target.split()))
        
        # Compute BLEU
        reference = [target.split()]
        candidate = prediction.split()
        bleu = sentence_bleu(reference, candidate)
        
        results.append({
            'bleu': bleu,
            'language': example.get('language', 'unknown'),
        })
    
    return results

