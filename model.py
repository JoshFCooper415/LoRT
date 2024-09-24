import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import math
from tqdm import tqdm
import flora_opt

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.u = nn.Parameter(torch.randn(in_features, rank, dtype=torch.float16) / (in_features ** 0.5))
        self.v = nn.Parameter(torch.randn(rank, out_features, dtype=torch.float16) / (rank ** 0.5))
        self.b = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
   
    def forward(self, x):
        # Ensure x is in float16
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
        
        # Low-rank factors for attention
        self.u_attn = nn.Parameter(torch.randn(n_heads, self.head_dim, rank, dtype=torch.float16) / (self.head_dim ** 0.5))
        self.v_attn = nn.Parameter(torch.randn(n_heads, rank, self.head_dim, dtype=torch.float16) / (rank ** 0.5))
        
        self.scale = 1.0 / math.sqrt(rank)
    
    def forward(self, x, mask=None):
        bs, seq_len, _ = x.size()
        
        qkv = self.qkv_linear(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(bs, seq_len, self.n_heads, self.head_dim).transpose(1, 2), qkv)
        
        # Low-rank attention computation
        q_low = torch.matmul(q, self.u_attn)  # Shape: [bs, n_heads, seq_len, rank]
        k_low = torch.matmul(k, self.u_attn)  # Shape: [bs, n_heads, seq_len, rank]
        
        scores = torch.matmul(q_low, k_low.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        v_low = torch.matmul(v, self.u_attn)  # Shape: [bs, n_heads, seq_len, rank]
        context_low = torch.matmul(attn, v_low)  # Shape: [bs, n_heads, seq_len, rank]
        context = torch.matmul(context_low, self.v_attn)  # Shape: [bs, n_heads, seq_len, head_dim]
        
        context = context.transpose(1, 2).contiguous().view(bs, seq_len, self.d_model)
        output = self.out(context)
        
        return output
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, rank, d_ff, dropout=0.1):
        super().__init__()
        self.attention = LowRankAttention(d_model, n_heads, rank, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            LowRankLinear(d_model, d_ff, rank),
            nn.ReLU(),
            LowRankLinear(d_ff, d_model, rank)
        )
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
    def __init__(self, vocab_size, d_model, n_heads, rank, n_layers, d_ff, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, rank, d_ff, dropout) for _ in range(n_layers)])
        
        self.fc_out = LowRankLinear(d_model, vocab_size, rank)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        x = self.dropout(self.pos_encoding(self.embedding(x)))
        
        for layer in self.layers:
            x = layer(x, mask)
        
        return self.fc_out(x)

# Hyperparameters
VOCAB_SIZE = 30_000
D_MODEL = 4096
N_HEADS = 32
RANK = 256
N_LAYERS = 32
D_FF = 1024
MAX_SEQ_LEN = 256
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1

# Load dataset
dataset = load_dataset("roneneldan/TinyStories")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False)

# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]["input_ids"], dtype=torch.long)

train_dataset = TinyStoriesDataset(tokenized_datasets["train"])
val_dataset = TinyStoriesDataset(tokenized_datasets["validation"])

# Collate function for DataLoader
def collate_fn(batch):
    # Pad sequences in the batch to the same length
    padded_batch = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=tokenizer.pad_token_id)
    return padded_batch

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LowRankTransformer(len(tokenizer), D_MODEL, N_HEADS, RANK, N_LAYERS, D_FF).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
optimizer = flora_opt.Flora(model.parameters(), lr=LEARNING_RATE)

# Training function
def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.
        total_tokens = 0
        progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for batch in progress_bar:
            input_ids = batch.to(device)
            
            # Create causal mask
            seq_len = input_ids.size(1)
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0)
            
            optimizer.zero_grad()
            output = model(input_ids, causal_mask)
            
            # Shift the targets for next token prediction
            shifted_output = output[:, :-1, :].contiguous()
            shifted_targets = input_ids[:, 1:].contiguous()
            
            # Create a mask for non-padding tokens
            non_pad_mask = (shifted_targets != tokenizer.pad_token_id).float()
            
            # Calculate loss only on non-padding tokens
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

# Evaluation function
def evaluate(model, dataloader, criterion):
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

# Parameter counting functions
def count_low_rank_parameters(model):
    total_params = 0
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            total_params += module.left.weight.numel() + module.right.weight.numel() + module.right.bias.numel()
        elif isinstance(module, nn.Embedding) or isinstance(module, nn.LayerNorm):
            total_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params

def count_full_rank_equivalent(model):
    total_params = 0
    for module in model.modules():
        if isinstance(module, LowRankLinear):
            total_params += module.left.in_features * module.right.out_features + module.right.out_features
        elif isinstance(module, nn.Embedding) or isinstance(module, nn.LayerNorm):
            total_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params

print(f"Using device: {device}")

# Train the model
train(model, train_dataloader, val_dataloader, criterion, optimizer, NUM_EPOCHS)

# Evaluate on validation set
val_loss = evaluate(model, val_dataloader, criterion)
print(f'Final Validation loss: {val_loss:.4f}')

# Save the model
torch.save(model.state_dict(), 'tiny_stories_low_rank_transformer.pth')
print("Model saved as 'tiny_stories_low_rank_transformer.pth'")

# Calculate and print the number of parameters
low_rank_params = count_low_rank_parameters(model)
full_rank_equivalent = count_full_rank_equivalent(model)

print(f"Number of parameters in the low-rank model: {low_rank_params:,}")
print(f"Equivalent number of parameters in a full-rank model: {full_rank_equivalent:,}")
print(f"Parameter reduction: {(1 - low_rank_params / full_rank_equivalent) * 100:.2f}%")