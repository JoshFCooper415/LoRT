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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.u = nn.Parameter(torch.randn(in_features, rank, dtype=torch.float16) / (in_features ** 0.5))
        self.v = nn.Parameter(torch.randn(rank, out_features, dtype=torch.float16) / (rank ** 0.5))
        self.b = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
   
    def forward(self, x):
        x = x.to(torch.float16)
        return torch.matmul(torch.matmul(x, self.u), self.v) + self.b

class LowRankAttention(nn.Module):
    def __init__(self, d_model, n_heads, rank, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_linear = LowRankLinear(d_model, 3 * d_model, rank)
        self.out = LowRankLinear(d_model, d_model, rank)
        self.dropout = nn.Dropout(dropout)
        
        self.u_attn = nn.Parameter(torch.randn(n_heads, self.head_dim, rank, dtype=torch.float16) / (self.head_dim ** 0.5))
        self.v_attn = nn.Parameter(torch.randn(n_heads, rank, self.head_dim, dtype=torch.float16) / (rank ** 0.5))
        
        self.scale = 1.0 / math.sqrt(rank)
    
    def forward(self, x, mask=None):
        bs, seq_len, _ = x.size()
        
        qkv = self.qkv_linear(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2), qkv)
        
        q_low = torch.matmul(q, self.u_attn)
        k_low = torch.matmul(k, self.u_attn)
        
        scores = torch.matmul(q_low, k_low.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        v_low = torch.matmul(v, self.u_attn)
        context_low = torch.matmul(attn, v_low)
        context = torch.matmul(context_low, self.v_attn)
        
        context = context.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        output = self.out(context)
        
        return output

class MixtureOfRanksLayer(nn.Module):
    def __init__(self, num_experts, input_size, output_size, hidden_size, rank):
        super().__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                LowRankLinear(input_size, hidden_size, rank),
                nn.ReLU(),
                LowRankLinear(hidden_size, output_size, rank)
            )
            for _ in range(num_experts)
        ])
        
        self.gate = nn.Linear(input_size, num_experts)
    
    def forward(self, x):
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, k=2, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        outputs = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                expert_inputs = x[mask]
                expert_outputs = self.experts[i](expert_inputs)
                expert_probs = top_k_probs[mask][torch.arange(mask.sum()), (top_k_indices[mask] == i).nonzero(as_tuple=True)[1]]
                outputs[mask] += expert_outputs * expert_probs.unsqueeze(-1)
        
        return outputs

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, num_experts, rank, d_ff, dropout=0.1):
        super().__init__()
        self.attention = LowRankAttention(d_model, n_heads, rank, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = MixtureOfRanksLayer(num_experts, d_model, d_model, d_ff, rank)
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

class LowRankTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_experts, rank, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, num_experts, rank, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.fc_out = LowRankLinear(d_model, vocab_size, rank)
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

def create_model_config(d_model, n_heads, num_experts, rank, n_layers, d_ff):
    return {
        "d_model": d_model,
        "n_heads": n_heads,
        "num_experts": num_experts,
        "rank": rank,
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
    model = LowRankTransformer(len(tokenizer), **model_config).to(device)
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

def run_rank_experiments(base_config, ranks, dataset_sizes, num_epochs, batch_size, learning_rate):
    results = []
    for rank in ranks:
        config = base_config.copy()
        config['rank'] = rank
        for size in dataset_sizes:
            print(f"Training model with rank {rank} and dataset size {size}")
            result = train_and_evaluate(config, size, num_epochs, batch_size, learning_rate)
            results.append(result)
            
            with open('MOR_rank_results.json', 'w') as f:
                json.dump(results, f, indent=2)

    return results
def plot_rank_analysis(results):
    plt.figure(figsize=(10, 6))
    
    dataset_sizes = sorted(set(r['dataset_size'] for r in results))
    ranks = sorted(set(r['model_config']['rank'] for r in results))
    
    for size in dataset_sizes:
        perplexities = [next(r['val_perplexity'] for r in results if r['dataset_size'] == size and r['model_config']['rank'] == rank) for rank in ranks]
        plt.plot(ranks, perplexities, marker='o', label=f'Dataset size: {size}')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank')
    plt.ylabel('Validation Perplexity')
    plt.title('Effect of Rank on Model Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig('MOR_rank_analysis.png')
    plt.close()

if __name__ == "__main__":
    base_config = create_model_config(256, 8, 8, None, 6, 256)  # rank will be set in the experiment loop
    ranks = [32, 64, 128]
    dataset_sizes = [10000, 50000, 100000, 500000]
    num_epochs = 1
    batch_size = 128
    learning_rate = 1e-4

    print("Starting rank analysis...")
    results = run_rank_experiments(base_config, ranks, dataset_sizes, num_epochs, batch_size, learning_rate)

    print("Plotting rank analysis...")
    plot_rank_analysis(results)

    print("Rank analysis complete. Results saved in 'rank_results.json' and plot saved as 'rank_analysis.png'.")