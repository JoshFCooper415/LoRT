import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
from tqdm import tqdm

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.u = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.v = nn.Parameter(torch.randn(rank, out_features) * 0.02)
        self.b = nn.Parameter(torch.zeros(out_features))
   
    def forward(self, x):
        return F.linear(F.linear(x, self.u), self.v) + self.b

class LowRankAttention(nn.Module):
    def __init__(self, d_model, n_heads, rank, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = LowRankLinear(d_model, d_model, rank)
        self.k_linear = LowRankLinear(d_model, d_model, rank)
        self.v_linear = LowRankLinear(d_model, d_model, rank)
        self.out = LowRankLinear(d_model, d_model, rank)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        q = self.q_linear(q).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        
        return self.out(context)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, rank, d_ff, dropout=0.1):
        super().__init__()
        self.attention = LowRankAttention(d_model, n_heads, rank, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            LowRankLinear(d_model, d_ff, rank),
            nn.GELU(),
            nn.Dropout(dropout),
            LowRankLinear(d_ff, d_model, rank)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attended = self.dropout1(self.attention(x, x, x, mask))
        x = self.norm1(x + attended)
        fedforward = self.dropout2(self.feed_forward(x))
        x = self.norm2(x + fedforward)
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
    def __init__(self, vocab_size, d_model, n_heads, rank, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, rank, d_ff, dropout) for _ in range(n_layers)])
        
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = LowRankLinear(d_model, vocab_size, rank)
        
        self._init_weights()
        
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x, mask=None):
        x = self.pos_encoding(self.embedding(x))
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        return self.fc_out(x)

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)

def collate_fn(batch):
    # Sort the batch in descending order of length
    batch.sort(key=lambda x: len(x), reverse=True)
    
    # Get the maximum sequence length in this batch
    max_len = len(batch[0])
    
    # Pad sequences
    padded_batch = torch.full((len(batch), max_len), 0, dtype=torch.long)  # Assuming 0 is the pad_token_id
    for i, seq in enumerate(batch):
        padded_batch[i, :len(seq)] = seq
    
    # Create attention mask
    attention_mask = (padded_batch != 0).float()
    
    return padded_batch, attention_mask

def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.
        total_tokens = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for batch, attention_mask in progress_bar:
            input_ids = batch.to(device)
            attention_mask = attention_mask.to(device)
            
            optimizer.zero_grad()
            output = model(input_ids, attention_mask)
            
            shifted_output = output[:, :-1, :].contiguous()
            shifted_targets = input_ids[:, 1:].contiguous()
            shifted_attention_mask = attention_mask[:, 1:].contiguous()
            
            loss = criterion(shifted_output.view(-1, output.size(-1)), shifted_targets.view(-1))
            loss = (loss * shifted_attention_mask.view(-1)).sum() / shifted_attention_mask.sum()
            
            if not torch.isfinite(loss):
                print("Warning: non-finite loss, ending training ")
                return
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_loss += loss.item() * shifted_attention_mask.sum().item()
            total_tokens += shifted_attention_mask.sum().item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / total_tokens
        val_loss = evaluate(model, val_dataloader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}')
        
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.
    total_tokens = 0
    with torch.no_grad():
        for batch, attention_mask in tqdm(dataloader, desc='Evaluating', leave=False):
            input_ids = batch.to(device)
            attention_mask = attention_mask.to(device)
            
            output = model(input_ids, attention_mask)
            
            shifted_output = output[:, :-1, :].contiguous()
            shifted_targets = input_ids[:, 1:].contiguous()
            shifted_attention_mask = attention_mask[:, 1:].contiguous()
            
            loss = criterion(shifted_output.view(-1, output.size(-1)), shifted_targets.view(-1))
            loss = (loss * shifted_attention_mask.view(-1)).sum() / shifted_attention_mask.sum()
            
            total_loss += loss.item() * shifted_attention_mask.sum().item()
            total_tokens += shifted_attention_mask.sum().item()
    
    return total_loss / total_tokens

def count_low_rank_parameters(model):
    total_params = 0
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            total_params += module.u.numel() + module.v.numel() + module.b.numel()
        elif isinstance(module, nn.Embedding) or isinstance(module, nn.LayerNorm):
            total_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params

def count_full_rank_equivalent(model):
    total_params = 0
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            total_params += module.u.size(0) * module.v.size(1) + module.b.numel()
        elif isinstance(module, nn.Embedding) or isinstance(module, nn.LayerNorm):
            total_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params

def prepare_data(dataset, tokenizer, max_seq_len, train_size=None):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_seq_len, padding=False)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    
    if train_size is not None:
        train_dataset = TinyStoriesDataset(tokenized_datasets["train"].select(range(train_size)))
    else:
        train_dataset = TinyStoriesDataset(tokenized_datasets["train"])
    
    val_dataset = TinyStoriesDataset(tokenized_datasets["validation"])
    
    return train_dataset, val_dataset

# Example usage (commented out)
'''
if __name__ == "__main__":
    # This is just an example of how to use the components defined above
    # You would typically import this file and use these components in another script

    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    train_dataset, val_dataset = prepare_data(dataset, tokenizer, max_seq_len=128, train_size=10000)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LowRankTransformer(
        vocab_size=len(tokenizer), 
        d_model=128, 
        n_heads=4, 
        rank=64, 
        n_layers=4, 
        d_ff=512
    ).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=3, device=device)

    # Evaluate the model
    val_loss = evaluate(model, val_dataloader, criterion, device)
    print(f"Final validation loss: {val_loss:.4f}")

    # Count parameters
    low_rank_params = count_low_rank_parameters(model)
    full_rank_equivalent = count_full_rank_equivalent(model)
    print(f"Low-rank parameters: {low_rank_params:,}")
    print(f"Full-rank equivalent: {full_rank_equivalent:,}")
    print(f"Parameter reduction: {(1 - low_rank_params / full_rank_equivalent) * 100:.2f}%")
'''