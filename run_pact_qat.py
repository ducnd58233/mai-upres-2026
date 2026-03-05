import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
import litert_torch

# Import your original unmodified model
from model_antsr import AntSR, RepConv, RepDWBlock, MobileOneRepBlock

# =====================================================================
# 1. The Differentiable PACT Activation Module
# =====================================================================
class PACTActivation(nn.Module):
    """
    Learnable Asymmetric Activation Clipping.
    Alpha and Beta are updated via backpropagation to find the mathematically
    optimal quantization range, ignoring severe outliers.
    """
    def __init__(self, qconfig, init_min=-2.0, init_max=2.0):
        super().__init__()
        # Learnable clipping bounds
        self.alpha = nn.Parameter(torch.tensor(init_max, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(init_min, dtype=torch.float32))
        
        # Instantiate PyTorch's FakeQuantize to sit immediately after the clip
        self.fq = qconfig.activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Perfectly differentiable clipping
        # y = clamp(x, beta, alpha)
        y = 0.5 * (torch.abs(x - self.beta) - torch.abs(x - self.alpha) + self.alpha + self.beta)
        
        # Apply QAT Fake Quantization. 
        # Because 'y' is strictly bounded by [beta, alpha], the FakeQuantize 
        # observer will naturally perfectly align with our learned parameters!
        return self.fq(y)

# =====================================================================
# 2. Dynamic Model Patcher (Monkey Patching)
# =====================================================================
def inject_pact_activations(model: nn.Module, qconfig):
    """
    Hunts down the hardcoded activations in AntSR and replaces them 
    with our learnable PACT activations.
    """
    for name, module in model.named_modules():
        # Inject into RepBlocks
        if isinstance(module, (RepConv, RepDWBlock, MobileOneRepBlock)):
            # Replace the standard 'self.act' (Identity or ReLU) with PACT
            module.act = PACTActivation(qconfig, init_min=-2.0, init_max=2.0)
            
    # Also replace the final output clamp (which usually defaults to MinClip 0-255)
    # We initialize it to [0.0, 255.0] but allow the network to refine it
    model.out_clamp = PACTActivation(qconfig, init_min=0.0, init_max=255.0)

# =====================================================================
# 3. Main PACT-QAT Training Loop
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32_ckpt", type=str, required=True, help="Path to Stage 2 FP32 ckpt")
    parser.add_argument("--out_tflite", type=str, required=True, help="Output INT8 TFLite path")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda_pact", type=float, default=1e-4, help="L2 penalty for PACT bounds")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Model & Load Stage 2 FP32 Weights
    print(f"Loading FP32 Model from {args.fp32_ckpt}...")
    model = AntSR(scale=3, channels=24, n_rep=6, rep_type="repdw", skip_mode="concat_raw").to(device)
    
    # Checkpoint loading logic (adapt based on how your team saves the dict)
    # ckpt = torch.load(args.fp32_ckpt, map_location=device)
    # model.load_state_dict(ckpt["model"])
    model.train()

    # 2. Setup QAT Config & INJECT PACT
    print("Injecting PACT Activations...")
    qconfig = tq.get_default_qat_qconfig('qnnpack')
    model.qconfig = qconfig
    
    # Apply monkey patch
    inject_pact_activations(model, qconfig)
    
    # Prepare the rest of the network
    tq.prepare_qat(model, inplace=True)

    # 3. Separate Optimizers for Weights and PACT Parameters
    # We often want the PACT bounds to learn slightly faster than the network weights
    pact_params = []
    net_params = []
    for name, param in model.named_parameters():
        if 'alpha' in name or 'beta' in name:
            pact_params.append(param)
        else:
            net_params.append(param)
            
    optimizer = torch.optim.Adam([
        {'params': net_params, 'lr': args.lr},
        {'params': pact_params, 'lr': args.lr * 10.0} # Learn bounds faster
    ])

    # ---------------------------------------------------------
    # 4. RUN STAGE 3 PACT-QAT TRAINING LOOP
    # ---------------------------------------------------------
    print("Running Stage 3 PACT-QAT Training...")
    
    for epoch in range(args.epochs): 
        # Dummy DataLoader (Replace with your actual DataLoader loop)
        dummy_lr = torch.rand(4, 3, 64, 64).to(device)
        dummy_hr = torch.rand(4, 3, 192, 192).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        sr = model(dummy_lr)
        
        # A. Standard Image Reconstruction Loss
        loss_l1 = F.l1_loss(sr, dummy_hr)
        
        # B. PACT Regularization Loss (Forces bounds to shrink and drop outliers)
        loss_pact = 0.0
        for p in pact_params:
            loss_pact += torch.sum(p ** 2)
            
        total_loss = loss_l1 + (args.lambda_pact * loss_pact)
        
        total_loss.backward()
        optimizer.step()

    print("PACT-QAT Training Complete.")

    # ---------------------------------------------------------
    # 5. DIRECT INT8 TFLITE EXPORT
    # ---------------------------------------------------------
    print("Exporting Native INT8 TFLite...")
    model.eval()
    model.cpu()
    
    model_quantized = tq.convert(model, inplace=False)
    
    example_input = torch.randn(1, 3, 720, 1280)
    quant_config = litert_torch.quantization.QuantConfig(
        quantization_mode=litert_torch.quantization.QuantizationMode.QAT
    )
    edge_model = litert_torch.convert(model_quantized, (example_input,), quant_config=quant_config)
    edge_model.export(args.out_tflite)
    print(f"Saved optimal model to {args.out_tflite}")

if __name__ == "__main__":
    main()