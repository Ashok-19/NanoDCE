import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import model_student
import Myloss
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class FeatureExtractor:
    
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        
        # Define which layers to extract features from
        # These should be the same layers in both teacher and student
        self.target_layers = {
            'e_conv2': 'conv2_features',
            'e_conv4': 'conv4_features', 
            'e_conv6': 'conv6_features'
        }
    
    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                hook = module.register_forward_hook(self.save_features(name))
                self.hooks.append(hook)
    


    def save_features(self, layer_name):
        def hook(module, input, output):
            feature_name = self.target_layers[layer_name]
            self.features[feature_name] = output
        return hook
    

    def clear_features(self):
        self.features = {}
    

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def knowledge_distillation_loss(student_output, teacher_output, temperature=4.0):
    
    
    mse_loss = nn.MSELoss()
    
   
    pixel_loss = mse_loss(student_output[0], teacher_output[0])
    
    illumination_loss = mse_loss(student_output[1], teacher_output[1])
    
    distillation_loss = pixel_loss + 0.1 * illumination_loss
    
    return distillation_loss


def feature_matching_loss(student_features, teacher_features):
    
    mse_loss = nn.MSELoss()
    feature_loss = 0.0
    
    # Define weights for different layers (earlier layers less important)
    layer_weights = {
        'conv2_features': 0.2,
        'conv4_features': 0.3, 
        'conv6_features': 0.5
    }
    
    # Compute feature matching loss for each layer
    for feature_name in student_features.keys():
        if feature_name in teacher_features:
            student_feat = student_features[feature_name]
            teacher_feat = teacher_features[feature_name].detach()  # Dont backprop to teacher
            
            # Use adaptive pooling to match spatial dimensions if needed
            if student_feat.shape[2:] != teacher_feat.shape[2:]:
                # Match spatial dimensions
                teacher_feat = nn.functional.adaptive_avg_pool2d(teacher_feat, student_feat.shape[2:])
            
            # Handle channel dimension mismatch by using only spatial information
            # Global average pooling to reduce to spatial information only
            if student_feat.shape[1] != teacher_feat.shape[1]:
                
                student_spatial = torch.mean(student_feat, dim=1, keepdim=True)  # [B, 1, H, W]
                teacher_spatial = torch.mean(teacher_feat, dim=1, keepdim=True)  # [B, 1, H, W]
                
                # Match spatial dimensions if still different
                if student_spatial.shape[2:] != teacher_spatial.shape[2:]:
                    teacher_spatial = nn.functional.adaptive_avg_pool2d(teacher_spatial, student_spatial.shape[2:])
                
                student_feat = student_spatial
                teacher_feat = teacher_spatial
            
            # Weighted feature loss
            weight = layer_weights.get(feature_name, 0.33)
            feature_loss += weight * mse_loss(student_feat, teacher_feat)
    
    return feature_loss


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Initialize Teacher Model
    scale_factor = config.scale_factor
    teacher_net = model.enhance_net_nopool(scale_factor).cuda()
    teacher_net.load_state_dict(torch.load(config.teacher_model_path))
    teacher_net.eval()  # Teacher in evaluation mode
    
    # Setup feature extraction for teacher
    teacher_extractor = FeatureExtractor(teacher_net)
    teacher_extractor.register_hooks()
    
    # Initialize Student Model
    student_net = model_student.enhance_net_nopool_student(scale_factor).cuda()
    
    # Setup feature extraction for student
    student_extractor = FeatureExtractor(student_net)
    student_extractor.register_hooks()
    
    if config.load_pretrain_student:
        student_net.load_state_dict(torch.load(config.pretrain_student_dir))
    
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)		
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, 
                                              shuffle=True, num_workers=config.num_workers, pin_memory=True)

    # Loss functions
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16)
    L_TV = Myloss.L_TV()

    optimizer = torch.optim.Adam(student_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    student_net.train()

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        epoch_distill_loss = 0.0
        epoch_feature_loss = 0.0
        epoch_original_loss = 0.0
        num_batches = 0
        
        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()

            # Clear previous features
            teacher_extractor.clear_features()
            student_extractor.clear_features()

            # Get teacher outputs (these are the targets) with features
            with torch.no_grad():
                teacher_enhanced_image, teacher_A = teacher_net(img_lowlight)
                teacher_features = teacher_extractor.features.copy()

            # Get student outputs with features
            student_enhanced_image, student_A = student_net(img_lowlight)
            student_features = student_extractor.features.copy()
            
            # Original losses for student
            Loss_TV = 1600 * L_TV(student_A)
            loss_spa = torch.mean(L_spa(student_enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(student_enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(student_enhanced_image, 0.6))
            original_loss = Loss_TV + loss_spa + loss_col + loss_exp

            # Knowledge distillation loss (output matching)
            distillation_loss = knowledge_distillation_loss(
                (student_enhanced_image, student_A),
                (teacher_enhanced_image, teacher_A)
            )
            
            # Feature matching loss
            feature_loss = feature_matching_loss(student_features, teacher_features)
            
            # Combined loss with three components
            loss = (config.alpha * distillation_loss +           
                   config.beta * feature_loss +                  
                   (1 - config.alpha - config.beta) * original_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_distill_loss += distillation_loss.item()
            epoch_feature_loss += feature_loss.item()
            epoch_original_loss += original_loss.item()
            num_batches += 1

            if ((iteration+1) % config.display_iter) == 0:
                print(f'Epoch {epoch+1}, Iteration {iteration+1}')
                print("Total Loss:", loss.item())
                print("Distillation Loss:", distillation_loss.item())
                print("Feature Loss:", feature_loss.item())
                print("Original Loss:", original_loss.item())
                
        # Save model
        if not os.path.exists(config.snapshots_folder):
            os.mkdir(config.snapshots_folder)
            
        avg_epoch_loss = epoch_loss / num_batches
        avg_distill_loss = epoch_distill_loss / num_batches
        avg_feature_loss = epoch_feature_loss / num_batches
        avg_original_loss = epoch_original_loss / num_batches
        
        print(f'Epoch {epoch+1} completed.')
        print(f"  Average Total Loss: {avg_epoch_loss:.4f}")
        print(f"  Average Distillation Loss: {avg_distill_loss:.4f}")
        print(f"  Average Feature Loss: {avg_feature_loss:.4f}")
        print(f"  Average Original Loss: {avg_original_loss:.4f}")
        
        torch.save(student_net.state_dict(), 
                  config.snapshots_folder + f"Student_Epoch{epoch+1}.pth") 		

    # Save final model
    torch.save(student_net.state_dict(), config.snapshots_folder + "Student_Final.pth")
    
    # Clean up hooks
    teacher_extractor.remove_hooks()
    student_extractor.remove_hooks()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--teacher_model_path', type=str, default="snapshots_Zero_DCE++/Epoch99.pth")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_Student_KD/")
    parser.add_argument('--load_pretrain_student', type=bool, default=False)
    parser.add_argument('--pretrain_student_dir', type=str, default="")
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for distillation loss')
    parser.add_argument('--beta', type=float, default=0.2, help='Weight for feature matching loss')
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for soft targets')

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)