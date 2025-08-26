# teacher.py (Enhanced Knowledge Distillation Pipeline - Fixed)
import torch
import torch.nn as nn
import torch.optim
# Import the scheduler module
import torch.optim.lr_scheduler as lr_scheduler
import os
import argparse
import dataloader
import zerodce
import model_student
import Myloss
import numpy as np
from torchvision import transforms
import shutil
import json
import torch.nn.functional as F
import random
import copy
import kd_loss
import Feature_extractor
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def validate_model(model, val_loader, device):
    """Simple validation function"""
    model.eval()
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for img_lowlight in val_loader:
            img_lowlight = img_lowlight.to(device)
            enhanced_image, _ = model(img_lowlight)
            mse = nn.functional.mse_loss(enhanced_image, img_lowlight, reduction='sum')
            total_mse += mse.item()
            total_samples += img_lowlight.size(0)

    model.train()
    avg_mse = total_mse / total_samples
    return avg_mse


def train_single_config(config, alpha, beta, run_name):
    """Train model with specific alpha and beta values"""
    print(f"\n{'='*60}")
    print(f"Training Student with Enhanced Knowledge Distillation Pipeline")
    print(f"  Alpha (Distillation): {alpha}, Beta (Feature): {beta}")
    print(f"{'='*60}")

    # Create run-specific snapshot folder
    run_snapshots_folder = os.path.join(config.snapshots_folder, run_name)
    if not os.path.exists(run_snapshots_folder):
        os.makedirs(run_snapshots_folder)

    # Initialize Teacher Model (ZeroDCE)
    teacher_net = zerodce.enhance_net_nopool().cuda()
    teacher_net.load_state_dict(torch.load(config.teacher_model_path))
    teacher_net.eval()

    FeatureExtractor = Feature_extractor()
    # Setup feature extraction for teacher
    teacher_extractor = FeatureExtractor(teacher_net, is_student=False)
    teacher_extractor.register_hooks()

    # Initialize Student Model
    student_net = model_student.enhance_net_nopool_student(config.scale_factor).cuda()
    
    # Setup EMA model for dynamic contrastive KD
    ema_student = copy.deepcopy(student_net)
    ema_alpha = 0.999  # Research-suggested value for EMA

    def update_ema():
        for ema_param, param in zip(ema_student.parameters(), student_net.parameters()):
            ema_param.data = ema_alpha * ema_param.data + (1 - ema_alpha) * param.data

    # Setup feature extraction for student
    student_extractor = FeatureExtractor(student_net, is_student=True)
    student_extractor.register_hooks()
    
    # Setup feature extraction for EMA model
    ema_extractor = FeatureExtractor(ema_student, is_student=True)
    ema_extractor.register_hooks()

    if config.load_pretrain_student:
        student_net.load_state_dict(torch.load(config.pretrain_student_dir))

    # Create dataset with gamma augmentation for multi-stage training
    class GammaAugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset, apply_gamma=False):
            self.base_dataset = base_dataset
            self.apply_gamma = apply_gamma
            
        def __len__(self):
            return len(self.base_dataset)
            
        def __getitem__(self, idx):
            img = self.base_dataset[idx]
            if self.apply_gamma:
                # Apply random gamma augmentation (0.5-1.5 range as suggested)
                gamma = random.uniform(0.5, 1.5)
                img = img ** gamma
            return img

    # STAGE 1: Self-supervised pretraining with gamma augmentation
    if config.stage1_epochs > 0:
        print(f"Starting Stage 1: Self-supervised pretraining ({config.stage1_epochs} epochs)")
        # Create dataset with gamma augmentation
        stage1_dataset = GammaAugmentedDataset(
            dataloader.lowlight_loader(config.lowlight_images_path), 
            apply_gamma=True
        )
        
        # Create validation split
        val_size = max(1, len(stage1_dataset) // 10)
        train_size = len(stage1_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(
            stage1_dataset, [train_size, val_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=config.train_batch_size,
            shuffle=True, num_workers=config.num_workers, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_subset, batch_size=config.train_batch_size,
            shuffle=False, num_workers=config.num_workers, pin_memory=True
        )

        # Optimizer for pretraining
        optimizer = torch.optim.Adam(
            student_net.parameters(), 
            lr=config.lr, 
            weight_decay=config.weight_decay
        )
        
        # Scheduler for pretraining
        if config.lr_scheduler_type == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.stage1_epochs, eta_min=config.lr_min
            )
        else:
            scheduler = None

        student_net.train()
        for epoch in range(config.stage1_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Stage 1 - Epoch {epoch+1}/{config.stage1_epochs}, Learning Rate: {current_lr:.6f}")

            for iteration, img_lowlight in enumerate(train_loader):
                img_lowlight = img_lowlight.cuda()

                # Get student outputs
                student_enhanced_image, student_A = student_net(img_lowlight)

                # Self-supervised loss (only exposure and TV)
                loss_multi_exposure = 7.0 * Myloss.MultiRegionExposureLoss(
                    patch_size=8,
                    dark_target=0.95,
                    mid_target=0.4,
                    bright_target_factor=0.6,  # Reduced to 0.65 (within 0.6-0.7 range)
                    bright_transition_width=0.12,  # Increased to 0.12 as suggested
                    dark_threshold=0.12,
                    mid_threshold=0.4,
                    bright_threshold=0.8
                )(student_enhanced_image, img_lowlight)
                
                Loss_TV = 200 * Myloss.L_TV()(student_A)
                original_loss = loss_multi_exposure + Loss_TV

                optimizer.zero_grad()
                original_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student_net.parameters(), config.grad_clip_norm
                )
                optimizer.step()

                epoch_loss += original_loss.item()
                num_batches += 1

                if ((iteration+1) % config.display_iter) == 0:
                    print(f'Stage 1 - Epoch {epoch+1}, Iteration {iteration+1}')
                    print(f"  Loss: {original_loss.item():.4f}")
                    print(f"  Gradient Norm: {grad_norm.item():.4f}")

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Validation
            val_loss = validate_model(student_net, val_loader, torch.device('cuda'))
            print(f'Stage 1 - Epoch {epoch+1} completed. Validation Loss: {val_loss:.6f}')
            
            # Save pretraining checkpoint
            torch.save(
                student_net.state_dict(),
                os.path.join(run_snapshots_folder, f"Stage1_Epoch{epoch+1}.pth")
            )

        # Update EMA after pretraining
        update_ema()
        print("Stage 1 completed. Moving to Stage 2...")

    # STAGE 2: KD fine-tuning
    print(f"Starting Stage 2: Knowledge distillation fine-tuning ({config.stage2_epochs} epochs)")
    
    # Create dataset without gamma augmentation (or with different augmentation)
    stage2_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    
    # Create validation split
    val_size = max(1, len(stage2_dataset) // 10)
    train_size = len(stage2_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        stage2_dataset, [train_size, val_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=config.train_batch_size,
        shuffle=True, num_workers=config.num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=config.train_batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True
    )

    # --- Original Loss functions from Myloss ---
    L_color_orig = Myloss.L_color()
    L_spa_orig = Myloss.L_spa()
    L_TV = Myloss.L_TV()
    L_Sa = Myloss.Sa_Loss()
    MultiRegionExposureLoss = Myloss.MultiRegionExposureLoss()

    # Multi-Region Exposure Loss with refined parameters
    L_multi_exposure = MultiRegionExposureLoss(
        patch_size=8,
        dark_target=0.95,
        mid_target=0.4,
        bright_target_factor=0.65,  # Reduced to 0.65 (within 0.6-0.7 range)
        bright_transition_width=0.12,  # Increased to 0.12 as suggested
        dark_threshold=0.12,
        mid_threshold=0.4,
        bright_threshold=0.8
    )

    # --- Optimizer ---
    optimizer = torch.optim.Adam(
        student_net.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    
    # --- Learning Rate Scheduler ---
    # Choose one scheduler based on config.lr_scheduler_type
    if config.lr_scheduler_type == "cosine":
        # Cosine Annealing: Reduces LR to eta_min over T_max epochs
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.stage2_epochs, eta_min=config.lr_min
        )
        print(f"Using CosineAnnealingLR scheduler. T_max={config.stage2_epochs}, eta_min={config.lr_min}")
        use_val_loss_for_scheduler = False  # Cosine doesn't use val loss
    elif config.lr_scheduler_type == "step":
        # Step Decay: Multiplies LR by gamma every step_size epochs
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
        )
        print(f"Using StepLR scheduler. step_size={config.lr_step_size}, gamma={config.lr_gamma}")
        use_val_loss_for_scheduler = False  # Step doesn't use val loss
    elif config.lr_scheduler_type == "plateau":
        # Reduce on Plateau: Reduces LR when val loss plateaus
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config.lr_gamma, 
            patience=config.lr_patience, verbose=True, min_lr=config.lr_min
        )
        print(f"Using ReduceLROnPlateau scheduler. factor={config.lr_gamma}, "
              f"patience={config.lr_patience}, min_lr={config.lr_min}")
        # Flag to indicate this scheduler needs validation loss
        use_val_loss_for_scheduler = True
    else:
        print(f"Warning: Unknown lr_scheduler_type '{config.lr_scheduler_type}'. No scheduler will be used.")
        scheduler = None
        use_val_loss_for_scheduler = False

    # STEP 1: Initialize VGG perceptual loss ONCE 
    VGGPerceptualLoss = Myloss.VGGPerceptualLoss()
    vgg_perceptual_loss = VGGPerceptualLoss().cuda()
    
    student_net.train()
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(config.stage2_epochs):
        epoch_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_feature_loss = 0.0
        epoch_original_loss = 0.0
        epoch_contrastive_loss = 0.0  # New contrastive loss component
        num_batches = 0
        
        # --- Print current learning rate ---
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Stage 2 - Epoch {epoch+1}/{config.stage2_epochs}, Current Learning Rate: {current_lr:.6f}")

        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()

            # Clear previous features
            teacher_extractor.clear_features()
            student_extractor.clear_features()
            ema_extractor.clear_features()

            # Get teacher outputs with features
            with torch.no_grad():
                teacher_enhanced_image, teacher_A = teacher_net(img_lowlight)
                teacher_features = teacher_extractor.features.copy()

            # Get student outputs with features
            student_enhanced_image, student_A = student_net(img_lowlight)
            student_features = student_extractor.features.copy()
            
            # Get EMA model outputs with features
            with torch.no_grad():
                ema_enhanced_image, ema_A = ema_student(img_lowlight)
                ema_features = ema_extractor.features.copy()

            # --- Original losses for student (Adjusted) ---
            Loss_TV = 200 * L_TV(student_A)  # Reduced from 1600
            loss_spa_orig = 1.5 * torch.mean(L_spa_orig(student_enhanced_image, img_lowlight))
            loss_col_orig = 1.2 * torch.mean(L_color_orig(student_enhanced_image))
            loss_sa = 0.3 * torch.mean(L_Sa(student_enhanced_image))
            loss_multi_exposure = 7.0 * L_multi_exposure(student_enhanced_image, img_lowlight)
            loss_denoise = L_TV(student_enhanced_image)

            # Combine all original and new losses
            original_loss = (
                Loss_TV +
                loss_spa_orig + loss_col_orig + loss_sa +
                loss_multi_exposure +
                loss_denoise
            )

            # STEP 2: Knowledge distillation loss - use the pre-initialized VGG
            distillation_loss = kd_loss.knowledge_distillation_loss(
                (student_enhanced_image, student_A),
                (teacher_enhanced_image, teacher_A),
                vgg_perceptual_loss=vgg_perceptual_loss  # Pass the pre-initialized instance
            )

            # Feature matching loss
            feature_loss = kd_loss.feature_matching_loss(student_features, teacher_features)
            
            # Dynamic contrastive loss component
            contrastive_loss = torch.tensor(0.0, device=img_lowlight.device)
            for name in student_features:
                if name in teacher_features and name in ema_features:
                    anchor = student_features[name]
                    positive = teacher_features[name].detach()
                    negative = ema_features[name].detach()
                    
                    pos_dist = F.l1_loss(anchor, positive)
                    neg_dist = F.l1_loss(anchor, negative)
                    # Dynamic contrastive term with weight 0.05 as suggested
                    contrastive_loss += 0.05 * (pos_dist / (neg_dist + 1e-7))
            
            # Update EMA after each step
            update_ema()

            # --- Combined loss ---
            effective_alpha = max(0.0, min(1.0, alpha))
            effective_beta = max(0.0, min(1.0 - effective_alpha, beta))
            remaining_weight = max(0.0, 1.0 - effective_alpha - effective_beta)

            if effective_alpha + effective_beta > 1.0:
                print(f"Warning: Adjusted alpha ({alpha}) + beta ({beta}) > 1.0. Normalizing.")
                total_weight = effective_alpha + effective_beta
                effective_alpha = effective_alpha / total_weight
                effective_beta = effective_beta / total_weight
                remaining_weight = 0.0

            loss = (effective_alpha * distillation_loss +
                    effective_beta * feature_loss +
                    remaining_weight * original_loss +
                    contrastive_loss)  # Add contrastive loss component

            optimizer.zero_grad()
            
            # --- CRITICAL FIX: Ensure 'loss' is a scalar tensor ---
            # Check if loss is a scalar (0-dimensional tensor)
            if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
                print(f"ERROR: 'loss' is not a scalar before .backward(). Type: {type(loss)}, Shape: {loss.shape if isinstance(loss, torch.Tensor) else 'N/A'}")
                print(f"  distillation_loss type/shape: {type(distillation_loss)}, {distillation_loss.shape if isinstance(distillation_loss, torch.Tensor) else 'N/A'}")
                print(f"  feature_loss type/shape: {type(feature_loss)}, {feature_loss.shape if isinstance(feature_loss, torch.Tensor) else 'N/A'}")
                print(f"  original_loss type/shape: {type(original_loss)}, {original_loss.shape if isinstance(original_loss, torch.Tensor) else 'N/A'}")
                # Force it to be a scalar by taking the mean (or sum)
                loss = torch.sum(loss)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_net.parameters(), config.grad_clip_norm
            )
            optimizer.step()

            epoch_loss += loss.item()
            epoch_distill_loss += distillation_loss.item()
            epoch_feature_loss += feature_loss.item()
            epoch_original_loss += original_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            num_batches += 1

            if ((iteration+1) % config.display_iter) == 0:
                print(f'Stage 2 - Epoch {epoch+1}, Iteration {iteration+1}')
                print("  Total Loss:", loss.item())
                print("  Distillation Loss:", distillation_loss.item())
                print("  Feature Loss:", feature_loss.item())
                print("  Contrastive Loss:", contrastive_loss.item())
                print("  Original Loss:", original_loss.item())
                print("    - TV Params:", Loss_TV.item())
                print("    - Spa (Orig):", loss_spa_orig.item())
                print("    - Col (Orig):", loss_col_orig.item())
                print("    - Sa (Orig)", loss_sa.item())
                print("    - Exp (Multi-Region):", loss_multi_exposure.item())
                print(f"  Gradient Norm: {grad_norm.item():.4f}")

        # Validation
        val_loss = validate_model(student_net, val_loader, torch.device('cuda'))
        
        # --- Update Learning Rate Scheduler ---
        if scheduler is not None:
            if use_val_loss_for_scheduler:
                # ReduceLROnPlateau needs the validation loss
                scheduler.step(val_loss)
            else:
                # Other schedulers update based on epoch
                scheduler.step()

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        avg_distill_loss = epoch_distill_loss / num_batches if num_batches > 0 else float('inf')
        avg_feature_loss = epoch_feature_loss / num_batches if num_batches > 0 else float('inf')
        avg_original_loss = epoch_original_loss / num_batches if num_batches > 0 else float('inf')
        avg_contrastive_loss = epoch_contrastive_loss / num_batches if num_batches > 0 else float('inf')

        print(f'Stage 2 - Epoch {epoch+1} completed.')
        print(f"  Average Total Loss: {avg_epoch_loss:.4f}")
        print(f"  Average Distillation Loss: {avg_distill_loss:.4f}")
        print(f"  Average Feature Loss: {avg_feature_loss:.4f}")
        print(f"  Average Contrastive Loss: {avg_contrastive_loss:.4f}")
        print(f"  Average Original Loss: {avg_original_loss:.4f}")
        print(f"  Validation Loss (MSE): {val_loss:.6f}")
        
        # Print updated LR after scheduler step
        if scheduler is not None:
            updated_lr = optimizer.param_groups[0]['lr']
            print(f"  Updated Learning Rate: {updated_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(
                student_net.state_dict(),
                os.path.join(run_snapshots_folder, "Best_Student.pth")
            )
            print(f"  -> New best model saved at epoch {best_epoch} with val loss {best_val_loss:.6f}")

        # Save checkpoint
        torch.save(
            student_net.state_dict(),
            os.path.join(run_snapshots_folder, f"Student_Epoch{epoch+1}.pth")
        )

    # Save final model
    torch.save(
        student_net.state_dict(), 
        os.path.join(run_snapshots_folder, "Student_Final.pth")
    )
    print(f"Final model saved to {os.path.join(run_snapshots_folder, 'Student_Final.pth')}")

    # Clean up hooks
    teacher_extractor.remove_hooks()
    student_extractor.remove_hooks()
    ema_extractor.remove_hooks()

    return best_val_loss, best_epoch

def grid_search(config):
    """Perform grid search over alpha and beta values"""
    alpha_values = [0.2, 0.3, 0.4]  # Much lower distillation weight
    beta_values = [0.05, 0.1, 0.15]  # Much lower feature matching weight

    results = []
    print("Starting Grid Search with Enhanced Knowledge Distillation Pipeline")
    print(f"Alpha values (Distillation weight): {alpha_values}")
    print(f"Beta values (Feature weight): {beta_values}")
    print(f"Total combinations: {len(alpha_values) * len(beta_values)}")

    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            run_name = f"alpha_{alpha}_beta_{beta}_enhanced_kd_pipeline"
            print(f"\nGrid search run {i*len(beta_values) + j + 1}/{len(alpha_values) * len(beta_values)}")

            try:
                val_loss, best_epoch = train_single_config(config, alpha, beta, run_name)

                result = {
                    'alpha': alpha,
                    'beta': beta,
                    'validation_loss': val_loss,
                    'best_epoch': best_epoch,
                    'run_name': run_name
                }
                results.append(result)

                print(f"Completed: alpha={alpha}, beta={beta}, val_loss={val_loss:.6f}")

            except Exception as e:
                print(f"Error in run alpha={alpha}, beta={beta}: {str(e)}")
                import traceback
                traceback.print_exc()
                result = {
                    'alpha': alpha,
                    'beta': beta,
                    'validation_loss': float('inf'),
                    'error': str(e),
                    'run_name': run_name
                }
                results.append(result)

    results.sort(key=lambda x: x['validation_loss'])

    results_file = os.path.join(config.snapshots_folder, "grid_search_results_enhanced_kd.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("GRID SEARCH RESULTS (Lowest validation loss first) - Enhanced KD Pipeline")
    print("="*70)

    for i, result in enumerate(results[:5]):
        if 'error' in result:
            print(f"{i+1}. Alpha: {result['alpha']}, Beta: {result['beta']} - ERROR: {result['error']}")
        else:
            print(f"{i+1}. Alpha: {result['alpha']}, Beta: {result['beta']} - "
                  f"Val Loss: {result['validation_loss']:.6f} (Best Epoch: {result['best_epoch']})")

    if results and 'error' not in results[0]:
        best_result = results[0]
        best_run_path = os.path.join(config.snapshots_folder, best_result['run_name'])
        best_model_path = os.path.join(best_run_path, "Best_Student.pth")

        if os.path.exists(best_model_path):
            overall_best_path = os.path.join(config.snapshots_folder, "Best_Student_Overall_Enhanced_KD.pth")
            shutil.copy2(best_model_path, overall_best_path)
            print(f"\nBest model (alpha={best_result['alpha']}, beta={best_result['beta']}) "
                  f"copied to main directory as Best_Student_Overall_Enhanced_KD.pth")

    return results

def train(config):
    """Main training function"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)

    if config.grid_search:
        return grid_search(config)
    else:
        train_single_config(config, config.alpha, config.beta, "single_run_enhanced_kd")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/lolv2/")
    parser.add_argument('--teacher_model_path', type=str, default="snapshots_Zero_DCE++/zerodce.pth")
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--stage1_epochs', type=int, default=10, 
                        help='Number of epochs for self-supervised pretraining')
    parser.add_argument('--stage2_epochs', type=int, default=20, 
                        help='Number of epochs for KD fine-tuning')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_student_enhanced_kd/")
    parser.add_argument('--load_pretrain_student', type=bool, default=False)
    parser.add_argument('--pretrain_student_dir', type=str, default="")
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for distillation loss')
    parser.add_argument('--beta', type=float, default=0.2, help='Weight for feature matching loss')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for soft targets')
    parser.add_argument('--grid_search', action='store_true', help='Run grid search instead of single training')
    
    # --- Learning Rate Scheduler Arguments ---
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine', 
                        choices=['cosine', 'step', 'plateau', 'none'], 
                        help='Type of learning rate scheduler')
    parser.add_argument('--lr_min', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--lr_step_size', type=int, default=15, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, 
                        help='Multiplicative factor for StepLR and ReduceLROnPlateau')
    parser.add_argument('--lr_patience', type=int, default=5, 
                        help='Patience for ReduceLROnPlateau scheduler')
    # --- End of Scheduler Arguments ---

    config = parser.parse_args()

    # Set stage2_epochs to num_epochs if not specified separately
    if config.stage2_epochs == 0:
        config.stage2_epochs = config.num_epochs
        config.stage1_epochs = 0

    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)

    train(config)