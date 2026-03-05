import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as tq
import litert_torch
from copy import deepcopy

# Import your original unmodified model
from model_antsr import AntSR

# =====================================================================
# 1. Non-Invasive Feature Extractor (Using PyTorch Hooks)
# =====================================================================
class HookBasedFeatureExtractor:
    """
    Attaches to a model and secretly copies the output of specified layers 
    during the forward pass without altering the model's source code.
    """
    def __init__(self, model: nn.Module, layer_names: list):
        self.features = {}
        self.hooks = []
        
        # Find the requested layers and attach hooks
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)

    def _get_hook(self, name: str):
        def hook(module, input, output):
            # Store the output tensor in our dictionary
            self.features[name] = output
        return hook

    def clear(self):
        """Must be called every iteration to prevent Memory (OOM) leaks."""
        self.features.clear()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

# =====================================================================
# 2. F-QKD Loss: Spatial Attention Transfer
# =====================================================================
def spatial_attention_loss(f_student: torch.Tensor, f_teacher: torch.Tensor, eps=1e-6):
    """
    Instead of forcing INT8 numbers to perfectly match FP32 (which is mathematically 
    impossible and causes instability), we force their *Spatial Attention* to match.
    We collapse the channels by squaring and summing, highlighting the "edges".
    """
    # 1. Compute spatial attention map: sum over channels (B, C, H, W) -> (B, 1, H, W)
    att_student = torch.sum(torch.abs(f_student) ** 2, dim=1, keepdim=True)
    att_teacher = torch.sum(torch.abs(f_teacher) ** 2, dim=1, keepdim=True)
    
    # 2. Flatten spatial dimensions (B, 1, H*W)
    att_student = att_student.view(att_student.size(0), -1)
    att_teacher = att_teacher.view(att_teacher.size(0), -1)
    
    # 3. L2 Normalize so we are comparing the *pattern*, not the absolute magnitude
    norm_student = F.normalize(att_student, p=2, dim=1, eps=eps)
    norm_teacher = F.normalize(att_teacher, p=2, dim=1, eps=eps)
    
    # 4. Compute L1 loss between the normalized attention maps
    return F.l1_loss(norm_student, norm_teacher)

# =====================================================================
# 3. Main F-QKD Training Loop
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize the Base Model
    print("Initializing Models...")
    model_params = dict(scale=3, channels=24, n_rep=6, rep_type="repdw", skip_mode="concat_raw")
    
    # Load Teacher (FP32 - mathematically perfect ceiling)
    teacher = AntSR(**model_params).to(device)
    # teacher.load_state_dict(torch.load("runs/antsr_sc/ckpt_stage2.pt")["model"])
    teacher.eval() # Teacher is frozen
    for param in teacher.parameters():
        param.requires_grad = False

    # Load Student (Will become INT8)
    student = AntSR(**model_params).to(device)
    # student.load_state_dict(torch.load("runs/antsr_sc/ckpt_stage2.pt")["model"])
    student.train()

    # 2. Prepare PyTorch QAT for the Student ONLY
    print("Preparing QAT for Student...")
    student.qconfig = tq.get_default_qat_qconfig('qnnpack')
    tq.prepare_qat(student, inplace=True)

    # 3. Setup Forward Hooks to intercept intermediate RepConv blocks
    # We will distill features from the 2nd, 4th, and 6th Rep blocks
    target_layers = ['rep.1', 'rep.3', 'rep.5']
    
    ext_teacher = HookBasedFeatureExtractor(teacher, target_layers)
    ext_student = HookBasedFeatureExtractor(student, target_layers)

    # 4. Optimization Setup
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-5)
    lambda_fqkd = 0.5 # Weight for the feature loss
    
    # ---------------------------------------------------------
    # 5. RUN STAGE 3 F-QKD TRAINING LOOP
    # ---------------------------------------------------------
    print("Running Stage 3 F-QKD Training...")
    
    # Dummy Training Loop (Replace with your actual DataLoader)
    for epoch in range(10): 
        dummy_lr = torch.rand(4, 3, 64, 64).to(device)
        dummy_hr = torch.rand(4, 3, 192, 192).to(device)
        
        optimizer.zero_grad()
        
        # A. Forward pass Teacher (No gradients)
        with torch.no_grad():
            teacher_out = teacher(dummy_lr)
            
        # B. Forward pass Student (With gradients and FakeQuant simulation)
        student_out = student(dummy_lr)
        
        # C. Calculate Output-Level Loss (Student vs Ground Truth / Teacher)
        loss_output = F.l1_loss(student_out, dummy_hr) # Standard SR loss
        loss_kd_out = F.l1_loss(student_out, teacher_out) # Output distillation
        
        # D. Calculate Feature-Level F-QKD Loss via Hooks
        loss_fqkd = 0.0
        for layer_name in target_layers:
            f_s = ext_student.features[layer_name]
            f_t = ext_teacher.features[layer_name]
            loss_fqkd += spatial_attention_loss(f_s, f_t)
            
        # Average the feature loss
        loss_fqkd = loss_fqkd / len(target_layers)
        
        # E. Total Loss and Backprop
        total_loss = loss_output + (0.1 * loss_kd_out) + (lambda_fqkd * loss_fqkd)
        total_loss.backward()
        optimizer.step()
        
        # F. CRITICAL: Clear the hook storage to prevent OOM
        ext_teacher.clear()
        ext_student.clear()
        
    print("F-QKD Training Complete.")
    ext_teacher.remove_hooks()
    ext_student.remove_hooks()

    # ---------------------------------------------------------
    # 6. DIRECT INT8 TFLITE EXPORT
    # ---------------------------------------------------------
    print("Exporting Native INT8 TFLite...")
    student.eval()
    student.cpu()
    
    # Convert QAT graph to evaluation graph
    model_quantized = tq.convert(student, inplace=False)
    
    # Direct export using litert_torch (bypassing PTQ)
    example_input = torch.randn(1, 3, 720, 1280)
    quant_config = litert_torch.quantization.QuantConfig(
        quantization_mode=litert_torch.quantization.QuantizationMode.QAT
    )
    edge_model = litert_torch.convert(model_quantized, (example_input,), quant_config=quant_config)
    edge_model.export("antsr_fqkd_int8.tflite")
    print("Saved optimal model to antsr_fqkd_int8.tflite")

if __name__ == "__main__":
    main()