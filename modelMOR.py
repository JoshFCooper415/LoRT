import torch
import torch.nn as nn
import torch.nn.functional as F

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.u = nn.Parameter(torch.randn(in_features, rank, dtype=torch.float16) / (in_features ** 0.5))
        self.v = nn.Parameter(torch.randn(rank, out_features, dtype=torch.float16) / (rank ** 0.5))
        self.b = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
   
    def forward(self, x):
        x = x.to(torch.float16)
        return torch.matmul(torch.matmul(x, self.u), self.v) + self.b

class MixedRankExpert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rank):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float16)
        self.fc2 = LowRankLinear(hidden_size, output_size, rank)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

class FullRankExpert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, dtype=torch.float16)
        self.fc2 = nn.Linear(hidden_size, output_size, dtype=torch.float16)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

class MixtureOfRanksLayer(nn.Module):
    def __init__(self, num_experts, input_size, hidden_size, output_size, rank):
        super().__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        num_low_rank = num_experts // 4
        num_full_rank = num_experts - num_low_rank
        
        self.experts = nn.ModuleList([
            MixedRankExpert(input_size, hidden_size, output_size, rank) if i < num_low_rank
            else FullRankExpert(input_size, hidden_size, output_size)
            for i in range(num_experts)
        ])
        
        self.gate = nn.Linear(input_size, num_experts, dtype=torch.float16)
    
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        
        # Compute gate logits and probabilities
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        print(f"Gate probabilities shape: {gate_probs.shape}")
        
        # Select top-k experts (let's use top-2 for this example)
        top_k_probs, top_k_indices = torch.topk(gate_probs, k=2, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities
        print(f"Top-k probabilities shape: {top_k_probs.shape}")
        print(f"Top-k indices shape: {top_k_indices.shape}")
        
        # Compute outputs from selected experts
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, self.output_size, dtype=torch.float16, device=x.device)
        print(f"Initial outputs shape: {outputs.shape}")
        
        for i in range(self.num_experts):
            mask = (top_k_indices == i).any(dim=-1)
            print(f"Mask shape for expert {i}: {mask.shape}")
            print(f"Number of True values in mask: {mask.sum().item()}")
            
            if mask.any():
                expert_inputs = x[mask]
                print(f"Expert inputs shape: {expert_inputs.shape}")
                
                expert_outputs = self.experts[i](expert_inputs)
                print(f"Expert outputs shape: {expert_outputs.shape}")
                
                # Corrected expert_probs selection
                expert_probs = top_k_probs[mask][torch.arange(mask.sum()), (top_k_indices[mask] == i).nonzero(as_tuple=True)[1]]
                print(f"Expert probs shape: {expert_probs.shape}")
                
                print(f"Outputs[mask] shape: {outputs[mask].shape}")
                print(f"Expert probs view shape: {expert_probs.view(-1, 1).shape}")
                
                try:
                    outputs[mask] += expert_outputs * expert_probs.view(-1, 1)
                except RuntimeError as e:
                    print(f"Error occurred: {str(e)}")
                    print(f"expert_outputs shape: {expert_outputs.shape}")
                    print(f"expert_probs.view(-1, 1) shape: {expert_probs.view(-1, 1).shape}")
                    raise e
        
        print(f"Final outputs shape: {outputs.shape}")
        return outputs
    
class MixtureOfRanksModel(nn.Module):
    def __init__(self, num_layers, num_experts, input_size, hidden_size, output_size, rank):
        super().__init__()
        self.layers = nn.ModuleList([
            MixtureOfRanksLayer(num_experts, input_size if i == 0 else hidden_size, hidden_size, 
                                hidden_size, rank)
            for i in range(num_layers - 1)
        ])
        self.layers.append(MixtureOfRanksLayer(num_experts, hidden_size, hidden_size, output_size, rank))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
input_size = 512
hidden_size = 2048
output_size = 512
num_experts = 8
num_layers = 4
rank = 64

model = MixtureOfRanksModel(num_layers, num_experts, input_size, hidden_size, output_size, rank)
x = torch.randn(32, input_size, dtype=torch.float16)  # Batch size of 32
output = model(x)
print(output.shape)  # Should be torch.Size([32, 512])