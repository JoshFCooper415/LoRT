import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
from tqdm import tqdm
import flora_opt
import json
import os
import matplotlib.pyplot as plt
import numpy as np

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.u = nn.Parameter(torch.randn(in_features, rank) / (in_features ** 0.5))
        self.v = nn.Parameter(torch.randn(rank, out_features) / (rank ** 0.5))
        self.b = nn.Parameter(torch.zeros(out_features))
   
    def forward(self, x):
        return torch.matmul(torch.matmul(x, self.u), self.v) + self.b
    
class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        bs, seq_len, _ = x.size()
        
        qkv = self.qkv_linear(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        output = self.out(context)
        
        return output

class ProgressiveRankMLP(nn.Module):
    def __init__(self, d_model, d_ff, ranks):
        super().__init__()
        self.layers = nn.ModuleList()
        in_features = d_model
        
        for i, rank in enumerate(ranks):
            out_features = d_ff if i < len(ranks) - 1 else d_model
            if rank == 'full':
                self.layers.append(nn.Linear(in_features, out_features))
            else:
                self.layers.append(LowRankLinear(in_features, out_features, rank))
            in_features = out_features
        
        self.activation = nn.ReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, ranks, dropout=0.1):
        super().__init__()
        self.attention = StandardAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = ProgressiveRankMLP(d_model, d_ff, ranks)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ProgressiveRankTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, ranks, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, ranks, dropout) 
            for _ in range(n_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.dropout(self.pos_encoding(self.embedding(x)))
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.fc_out(x)
    
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding=False)

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)

def collate_fn(batch):
    padded_batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    return padded_batch

def create_model_config(d_model, n_heads, ranks, n_layers, d_ff):
    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "ranks": ranks,
        "n_layers": n_layers,
        "d_ff": d_ff
    }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs):
    device = next(model.parameters()).device
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.
        total_tokens = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for batch in progress_bar:
            input_ids = batch.to(device)
            
            seq_len = input_ids.size(1)
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)
            
            optimizer.zero_grad()
            output = model(input_ids, causal_mask)
            
            shifted_output = output[:, :-1, :].contiguous()
            shifted_targets = input_ids[:, 1:].contiguous()
            
            non_pad_mask = (shifted_targets != tokenizer.pad_token_id).float()
            
            loss = criterion(shifted_output.view(-1, len(tokenizer)), shifted_targets.view(-1))
            loss = (loss * non_pad_mask.view(-1)).sum() / non_pad_mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * non_pad_mask.sum().item()
            total_tokens += non_pad_mask.sum().item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / total_tokens
        val_loss = evaluate(model, val_dataloader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}')

def evaluate(model, dataloader, criterion):
    device = next(model.parameters()).device
    model.eval()
    total_loss = 0.
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch.to(device)
            
            seq_len = input_ids.size(1)
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)
            
            output = model(input_ids, causal_mask)
            
            shifted_output = output[:, :-1, :].contiguous()
            shifted_targets = input_ids[:, 1:].contiguous()
            
            non_pad_mask = (shifted_targets != tokenizer.pad_token_id).float()
            
            loss = criterion(shifted_output.view(-1, len(tokenizer)), shifted_targets.view(-1))
            loss = (loss * non_pad_mask.view(-1)).sum() / non_pad_mask.sum()
            
            total_loss += loss.item() * non_pad_mask.sum().item()
            total_tokens += non_pad_mask.sum().item()
    
    return total_loss / total_tokens

def train_and_evaluate(model_config, dataset_size, num_epochs, batch_size, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ProgressiveRankTransformer(len(tokenizer), **model_config).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    optimizer = flora_opt.Flora(model.parameters(), lr=learning_rate)

    full_dataset = load_dataset("roneneldan/TinyStories")
    train_dataset = full_dataset["train"].select(range(dataset_size))
    val_dataset = full_dataset["validation"].select(range(min(10000, len(full_dataset["validation"]))))

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

    train_dataloader = DataLoader(TinyStoriesDataset(tokenized_train), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(TinyStoriesDataset(tokenized_val), batch_size=batch_size, collate_fn=collate_fn)

    train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs)

    val_loss = evaluate(model, val_dataloader, criterion)
    val_perplexity = math.exp(val_loss)

    return {
        "model_config": model_config,
        "dataset_size": dataset_size,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_parameters": count_parameters(model),
        "val_loss": val_loss,
        "val_perplexity": val_perplexity
    }

def run_rank_experiments(base_configs, dataset_sizes, num_epochs, batch_size, learning_rate):
    results = []
    for config in base_configs:
        for size in dataset_sizes:
            print(f"Training model with ranks {config['ranks']} and dataset size {size}")
            result = train_and_evaluate(config, size, num_epochs, batch_size, learning_rate)
            results.append(result)
            
            with open('progressive_rank_results.json', 'w') as f:
                json.dump(results, f, indent=2)

    return results

def plot_rank_analysis(results):
    plt.figure(figsize=(12, 8))
    
    dataset_sizes = sorted(set(r['dataset_size'] for r in results))
    rank_configs = [tuple(r['model_config']['ranks']) for r in results]
    unique_configs = sorted(set(rank_configs), key=lambda x: (x.count('full'), x))
    
    for size in dataset_sizes:
        perplexities = [next(r['val_perplexity'] for r in results if r['dataset_size'] == size and tuple(r['model_config']['ranks']) == config) for config in unique_configs]
        plt.plot(range(len(unique_configs)), perplexities, marker='o', label=f'Dataset size: {size}')
    
    plt.xlabel('Rank Configuration')
    plt.ylabel('Validation Perplexity')
    plt.title('Effect of Progressive Rank on Model Performance')
    plt.legend()
    plt.xticks(range(len(unique_configs)), [str(config) for config in unique_configs], rotation=45, ha='right')
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('progressive_rank_analysis.png')
    plt.close()

if __name__ == "__main__":
    base_configs = [
        create_model_config(256, 8, [32, 64, 128, 'full', 'full', 'full', 'full', 'full'], 6, 512),
        create_model_config(256, 8, ['full', 'full', 'full', 'full', 'full', 128, 64, 32], 6, 512),
        create_model_config(256, 8, ['full', 'full', 'full', 'full', 'full', 'full', 'full', 'full'], 6, 512),
    ]
    dataset_sizes = [10000, 50000, 100000, 500000]
    num_epochs = 1
    batch_size = 64
    learning_rate = 1e-4

    print("Starting progressive rank analysis...")
    results = run_rank_experiments(base_configs, dataset_sizes, num_epochs, batch_size, learning_rate)

    print("Plotting progressive rank analysis...")
    plot_rank_analysis(results)

    print("Progressive rank analysis complete. Results saved in 'progressive_rank_results.json' and plot saved as 'progressive_rank_analysis.png'.")