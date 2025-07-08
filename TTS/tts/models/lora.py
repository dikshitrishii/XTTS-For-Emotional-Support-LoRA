import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if r > 0 else 1
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora = self.dropout(x) @ self.lora_A.t()
            lora = lora @ self.lora_B.t()
            result = result + self.scale * lora
        return result

def apply_lora_to_gpt(gpt_module, r=8, alpha=16, dropout=0.05, target_keywords=("attn", "linear")):
    import torch.nn as nn
    from lora import LoRALinear  # Adjust import if needed

    for name, child in gpt_module.named_children():
        if isinstance(child, nn.Linear) and any(k in name.lower() for k in target_keywords):
            lora_linear = LoRALinear(child.in_features, child.out_features, r, alpha, dropout)
            lora_linear.weight.data = child.weight.data.clone()
            if child.bias is not None:
                lora_linear.bias.data = child.bias.data.clone()
            setattr(gpt_module, name, lora_linear)
        else:
            apply_lora_to_gpt(child, r, alpha, dropout, target_keywords)
    return gpt_module
