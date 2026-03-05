import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
import litert_torch  # For direct INT8 export

# Import your original unmodified model
from model_antsr import AntSR, RepConv

# =====================================================================
# 1. Differentiable Math Helpers (Overrides original @torch.no_grad)
# =====================================================================
def fuse_batchnorm_into_conv_diff(W: torch.Tensor, b: torch.Tensor, bn: nn.BatchNorm2d):
    """Differentiable BN fusion (no @torch.no_grad)"""
    inv_std = torch.rsqrt(bn.running_var + bn.eps)
    scale = bn.weight * inv_std
    W_fused = W * scale.view(-1, 1, 1, 1)
    b_fused = (b - bn.running_mean) * scale + bn.bias
    return W_fused, b_fused

def pad_1x1_to_3x3_center_diff(w_1x1: torch.Tensor):
    """Differentiable padding using F.pad"""
    return F.pad(w_1x1, (1, 1, 1, 1), value=0.0)

# =====================================================================
# 2. The RepQ-SR Wrapper Class
# =====================================================================
class RepConvQATWrapper(nn.Module):
    """
    Wraps an existing RepConv block. During QAT, it dynamically calculates
    the fused 3x3 kernel, applies FakeQuantize, and convolves.
    """
    def __init__(self, orig_block: RepConv, qconfig):
        super().__init__()
        self.block = orig_block # Hold the original parameters
        self.c = orig_block.c
        
        # Instantiate FakeQuantize nodes for the fused kernel and output
        self.weight_fq = qconfig.weight()
        self.act_fq = qconfig.activation()

        # Disable QAT on the inner components to prevent double-quantization
        for module in self.block.modules():
            module.qconfig = None

    def get_fused_weights_diff(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates W_eq and b_eq with gradients intact."""
        W1 = self.block.conv1_a.weight.squeeze(-1).squeeze(-1)   
        W3 = self.block.conv3.weight                              
        b3 = self.block.conv3.bias                                
        W2 = self.block.conv1_b.weight.squeeze(-1).squeeze(-1)   
        b2 = self.block.conv1_b.bias                              

        tmp = torch.einsum("mnxy,ni->mixy", W3, W1)         
        W_main = torch.einsum("om,mixy->oixy", W2, tmp)     
        b_main = b2 + torch.matmul(W2, b3)                  

        W_skip = pad_1x1_to_3x3_center_diff(self.block.conv1_s.weight)
        b_skip = self.block.conv1_s.bias

        W_eq = W_main + W_skip
        b_eq = b_main + b_skip

        if isinstance(self.block.bn, nn.BatchNorm2d):
            W_eq, b_eq = fuse_batchnorm_into_conv_diff(W_eq, b_eq, self.block.bn)

        return W_eq, b_eq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Differentiable Fusion
        W_eq, b_eq = self.get_fused_weights_diff()
        
        # 2. Quantize the FUSED weights
        W_q = self.weight_fq(W_eq)
        
        # 3. Convolve
        y = F.conv2d(x, W_q, b_eq, padding=1)
        
        # 4. Activate and Quantize Output
        y = self.block.act(y)
        return self.act_fq(y)

# =====================================================================
# 3. Dynamic Model Patcher
# =====================================================================
def inject_repq_wrappers(model: nn.Module, qconfig):
    """Recursively replaces RepConv blocks with RepConvQATWrapper."""
    for name, child in model.named_children():
        if isinstance(child, RepConv):
            # Replace the module with our wrapper
            wrapper = RepConvQATWrapper(child, qconfig)
            setattr(model, name, wrapper)
        else:
            # Recurse deeper into the network
            inject_repq_wrappers(child, qconfig)

# =====================================================================
# 4. Main Execution (Training + Direct Export)
# =====================================================================
def main():
    # 1. Load your trained FP32 Model (Stage 1+2 complete)
    print("Loading FP32 Model...")
    model = AntSR(scale=3, channels=24, n_rep=6, rep_type="repconv").cuda()
    # model.load_state_dict(torch.load("runs/antsr_sc/ckpt_stage2.pt")["model"])

    # 2. Setup PyTorch QNNPACK QAT
    qconfig = tq.get_default_qat_qconfig('qnnpack')
    model.qconfig = qconfig

    # 3. INJECT RepQ-SR WRAPPERS (Monkey Patching)
    print("Injecting RepQ-SR Wrappers...")
    inject_repq_wrappers(model, qconfig)

    # 4. Prepare the rest of the network (global adds, conv_in, conv_out)
    tq.prepare_qat(model, inplace=True)
    
    # ---------------------------------------------------------
    # 5. RUN STAGE 3 QAT TRAINING LOOP HERE
    # ---------------------------------------------------------
    print("Running Stage 3 RepQ-QAT...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    
    # Dummy training loop
    for epoch in range(10): # Replace with your actual dataloader
        dummy_lr = torch.rand(4, 3, 64, 64).cuda()
        dummy_hr = torch.rand(4, 3, 192, 192).cuda()
        
        optimizer.zero_grad()
        sr = model(dummy_lr)
        loss = F.l1_loss(sr, dummy_hr)
        loss.backward()
        optimizer.step()
    print("QAT Training Complete.")

    # ---------------------------------------------------------
    # 6. DIRECT INT8 TFLITE EXPORT (Bypasses PTQ Script!)
    # ---------------------------------------------------------
    print("Exporting Native INT8 TFLite...")
    model.eval()
    model.cpu()
    
    # Convert back from QAT to standard quantized evaluation graph
    model_quantized = tq.convert(model, inplace=False)
    
    # Export natively to TFLite utilizing the learned FakeQuant scales
    example_input = torch.randn(1, 3, 720, 1280)
    
    quant_config = litert_torch.quantization.QuantConfig(
        quantization_mode=litert_torch.quantization.QuantizationMode.QAT
    )
    
    edge_model = litert_torch.convert(
        model_quantized, 
        (example_input,), 
        quant_config=quant_config
    )
    
    edge_model.export("antsr_repq_int8.tflite")
    print("Saved optimal model to antsr_repq_int8.tflite")

if __name__ == "__main__":
    main()