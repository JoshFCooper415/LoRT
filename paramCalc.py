def calculate_params_vram_flops(vocab_size, d_model, n_heads, rank, n_layers, d_ff, max_seq_len, batch_size):
    def low_rank_linear_params(in_features, out_features, rank):
        return in_features * rank + rank * out_features + out_features
    def full_rank_linear_params(in_features, out_features):
        return in_features * out_features + out_features
    
    # Embedding layers (same for both versions)
    embedding_params = vocab_size * d_model  # Token embedding
    positional_embedding_params = max_seq_len * d_model  # Positional embedding
    
    # Transformer layers
    def transformer_layer_params(is_low_rank):
        if is_low_rank:
            # Low-rank attention
            qkv_params = 3 * low_rank_linear_params(d_model, d_model, rank)
            out_proj_params = low_rank_linear_params(d_model, d_model, rank)
            # Low-rank feed-forward
            ff_params = low_rank_linear_params(d_model, d_ff, rank) + low_rank_linear_params(d_ff, d_model, rank)
        else:
            # Full-rank attention
            qkv_params = 3 * full_rank_linear_params(d_model, d_model)
            out_proj_params = full_rank_linear_params(d_model, d_model)
            # Full-rank feed-forward
            ff_params = full_rank_linear_params(d_model, d_ff) + full_rank_linear_params(d_ff, d_model)
        # Layer normalization (same for both versions)
        layer_norm_params = 4 * d_model  # 2 layer norms per transformer layer, each with weight and bias
        return qkv_params + out_proj_params + ff_params + layer_norm_params
    
    # Total params for all transformer layers
    transformer_params_low_rank = n_layers * transformer_layer_params(is_low_rank=True)
    transformer_params_full_rank = n_layers * transformer_layer_params(is_low_rank=False)
    
    # Output layer
    output_layer_params_low_rank = low_rank_linear_params(d_model, vocab_size, rank)
    output_layer_params_full_rank = full_rank_linear_params(d_model, vocab_size)
    
    # Total params
    total_params_low_rank = (embedding_params + positional_embedding_params +
                             transformer_params_low_rank + output_layer_params_low_rank)
    total_params_full_rank = (embedding_params + positional_embedding_params +
                              transformer_params_full_rank + output_layer_params_full_rank)
    
    # VRAM usage calculation
    def calculate_vram_usage(total_params, batch_size, max_seq_len, d_model):
        param_memory = total_params * 4  # 4 bytes per parameter
        activation_memory = 2 * batch_size * max_seq_len * d_model * 4 * n_layers
        attention_memory = batch_size * n_heads * max_seq_len * max_seq_len * 4
        total_vram = param_memory + activation_memory + attention_memory
        return total_vram / (1024 ** 3)  # Convert to GB
    
    vram_usage_low_rank = calculate_vram_usage(total_params_low_rank, batch_size, max_seq_len, d_model)
    vram_usage_full_rank = calculate_vram_usage(total_params_full_rank, batch_size, max_seq_len, d_model)
    
    # FLOPs calculation
    def calculate_flops(is_low_rank):
        # Embedding lookup (negligible compared to other operations, so we'll ignore it)
        
        # Self-attention
        if is_low_rank:
            qkv_flops = 3 * 2 * d_model * rank * max_seq_len  # Low-rank projection
            qkv_flops += 3 * 2 * rank * d_model * max_seq_len  # Low-rank projection (second part)
        else:
            qkv_flops = 3 * 2 * d_model * d_model * max_seq_len  # Full-rank projection
        
        attn_flops = 2 * n_heads * max_seq_len * max_seq_len * (d_model // n_heads)  # Q * K^T
        attn_flops += n_heads * max_seq_len * max_seq_len * max_seq_len  # softmax and weighted sum
        
        if is_low_rank:
            out_proj_flops = 2 * d_model * rank * max_seq_len  # Low-rank projection
            out_proj_flops += 2 * rank * d_model * max_seq_len  # Low-rank projection (second part)
        else:
            out_proj_flops = 2 * d_model * d_model * max_seq_len  # Full-rank projection
        
        # Feed-forward network
        if is_low_rank:
            ff_flops = 2 * d_model * rank * max_seq_len  # First low-rank projection
            ff_flops += 2 * rank * d_ff * max_seq_len  # First low-rank projection (second part)
            ff_flops += 2 * d_ff * rank * max_seq_len  # Second low-rank projection
            ff_flops += 2 * rank * d_model * max_seq_len  # Second low-rank projection (second part)
        else:
            ff_flops = 2 * d_model * d_ff * max_seq_len  # First projection
            ff_flops += 2 * d_ff * d_model * max_seq_len  # Second projection
        
        # Layer normalization (simplified)
        layer_norm_flops = 2 * d_model * max_seq_len
        
        # Total FLOPs for one layer
        layer_flops = qkv_flops + attn_flops + out_proj_flops + ff_flops + 2 * layer_norm_flops
        
        # Total FLOPs for all layers
        total_flops = layer_flops * n_layers
        
        # Output layer (we'll use the same computation as qkv_flops)
        if is_low_rank:
            total_flops += 2 * d_model * rank * max_seq_len + 2 * rank * vocab_size * max_seq_len
        else:
            total_flops += 2 * d_model * vocab_size * max_seq_len
        
        return total_flops
    
    flops_low_rank = calculate_flops(is_low_rank=True)
    flops_full_rank = calculate_flops(is_low_rank=False)
    
    return {
        "Low-Rank Model": {
            "Total Params": total_params_low_rank,
            "Trainable Params": total_params_low_rank,  # All params are trainable
            "VRAM Usage (GB)": vram_usage_low_rank,
            "FLOPs per inference": flops_low_rank
        },
        "Full-Rank Model": {
            "Total Params": total_params_full_rank,
            "Trainable Params": total_params_full_rank,  # All params are trainable
            "VRAM Usage (GB)": vram_usage_full_rank,
            "FLOPs per inference": flops_full_rank
        }
    }

# Hyperparameters
VOCAB_SIZE = 30000
D_MODEL = 4096
N_HEADS = 16
RANK = 32
N_LAYERS = 16
D_FF = 2048
MAX_SEQ_LEN = 512
BATCH_SIZE = 32  # Added batch size for VRAM calculation

# Calculate parameters, VRAM usage, and FLOPs
results = calculate_params_vram_flops(VOCAB_SIZE, D_MODEL, N_HEADS, RANK, N_LAYERS, D_FF, MAX_SEQ_LEN, BATCH_SIZE)

# Print results
for model_type, params in results.items():
    print(f"\n{model_type}:")
    for param_type, count in params.items():
        if isinstance(count, int):
            print(f"  {param_type}: {count:,}")
        else:
            print(f"  {param_type}: {count:.2f}")

# Calculate and print the reductions
low_rank_params = results["Low-Rank Model"]["Total Params"]
full_rank_params = results["Full-Rank Model"]["Total Params"]
low_rank_vram = results["Low-Rank Model"]["VRAM Usage (GB)"]
full_rank_vram = results["Full-Rank Model"]["VRAM Usage (GB)"]
low_rank_flops = results["Low-Rank Model"]["FLOPs per inference"]
full_rank_flops = results["Full-Rank Model"]["FLOPs per inference"]

reduction_percentage = (1 - low_rank_params / full_rank_params) * 100
vram_reduction_percentage = (1 - low_rank_vram / full_rank_vram) * 100
flops_reduction_percentage = (1 - low_rank_flops / full_rank_flops) * 100

print(f"\nParameter reduction: {reduction_percentage:.2f}%")
print(f"VRAM usage reduction: {vram_reduction_percentage:.2f}%")
print(f"FLOPs reduction: {flops_reduction_percentage:.2f}%")