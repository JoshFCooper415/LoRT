import torch
import torch.nn as nn
from transformers import AutoTokenizer
from model import LowRankTransformer, D_MODEL, N_HEADS, RANK, N_LAYERS, D_FF

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load the trained model
def load_model(model_path):
    model = LowRankTransformer(len(tokenizer), D_MODEL, N_HEADS, RANK, N_LAYERS, D_FF).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Generate a story
def generate_story(model, prompt="Once upon a time", max_length=100):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            seq_len = input_ids.size(1)
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)
            
            outputs = model(input_ids, causal_mask)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load the trained model
    model_path = "tiny_stories_low_rank_transformer.pth"
    model = load_model(model_path)
    
    # Generate and print a story
    prompt = "Once upon a time, in a magical forest"
    generated_story = generate_story(model, prompt)
    print(f"Generated Story:\n{generated_story}")
    
    # Interactive story generation
    print("\nInteractive Story Generation")
    print("Enter a prompt to generate a story. Type 'quit' to exit.")
    while True:
        user_prompt = input("\nEnter your prompt: ")
        if user_prompt.lower() == 'quit':
            break
        generated_story = generate_story(model, user_prompt)
        print(f"\nGenerated Story:\n{generated_story}")

    print("Thank you for using the story generator!")