**This file explains the training of the student model using knowledge distillation**

## Knowledge Distillation Pipeline

  We employ a two-stage training pipeline inspired by self-supervised pretraining and feature-aligned KD, tailored for zero-reference enhancement. ZeroDCE++ is chosen as the teacher model.
  
### Stage 1: Self-Supervised Pretraining

**Objective**: Initialize the student independently to avoid early over-reliance on the teacher, mitigating the "moving target" issue in KD.
**Augmentation**: Gamma correction (random γ ∈ [0.5, 1.5]) simulates varied exposures on low-light images.
**Losses**: Multi-Region Exposure Loss (aggressive boosting for dark regions, suppression for bright) + Total Variation (TV) smoothness on curve parameters.
*Duration*: Configurable (default: 40 epochs).
*Rationale*: Draws from Zero-DCE's non-reference losses

### Stage 2: KD Fine-Tuning

Teacher Model: Pre-trained Zero-DCE++ loaded and frozen. Provides stable references, preventing co-adaptation instabilities
Student Model: Simplified architecture (see below).

### Distillation Components:

**_Output Alignment (Hard Logits)_**: MSE on enhanced images and illumination maps. Hard logits are used (no temperature softening) as enhancement is a continuous regression task, unlike classification where softmax+temperature distills "dark knowledge"

**_Perceptual (VGG-based)_** and **_SSIM losses_** added for structural fidelity.
Feature Matching: Extracts features from teacher's e_conv2, e_conv4, e_conv7 and aligns with student's corresponding layers (e_conv1 → e_conv2, etc.). Handles mismatches via adaptive pooling (spatial) and GAP (channels). Weighted by layer depth (0.2 early, 0.3 mid, 0.5 deep) to emphasize hierarchical knowledge transfer

**_Contrastive Loss_**: Uses EMA teacher for negative samples, encouraging student to pull closer to fixed teacher while pushing away from its EMA version (inspired by BYOL/MoCo for stability).

**_Task-Specific Losses_**: From Zero-DCE++ (color constancy, spatial consistency, TV smoothness, multi-region exposure) to maintain enhancement constraints.

**Combined Loss**: 
        
        _α × distillation + β × feature_matching + (1 - α - β) × task_losses + contrastive_loss

        _Defaults_: α=0.3–0.4, β=0.1 (tuned via grid search to balance mimicry and independence).

