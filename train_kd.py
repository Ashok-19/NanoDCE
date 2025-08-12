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


def knowledge_distillation_loss(student_output, teacher_output, original_input, temperature=4.0, alpha=0.7):
    """
    Compute knowledge distillation loss
    """
    # MSE loss between student and teacher outputs
    mse_loss = nn.MSELoss()
    
    # Pixel-wise loss (student should match teacher's enhanced image)
    pixel_loss = mse_loss(student_output[0], teacher_output[0])
    
    # Illumination map loss
    illumination_loss = mse_loss(student_output[1], teacher_output[1])
    
    # Total distillation loss
    distillation_loss = pixel_loss + 0.1 * illumination_loss
    
    return distillation_loss


def train(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Initialize Teacher Model
    scale_factor = config.scale_factor
    teacher_net = model.enhance_net_nopool(scale_factor).cuda()
    teacher_net.load_state_dict(torch.load(config.teacher_model_path))
    teacher_net.eval()  # Teacher in evaluation mode
    
    # Initialize Student Model
    student_net = model_student.enhance_net_nopool_student(scale_factor).cuda()
    
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
        num_batches = 0
        
        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()

            # Get teacher outputs (these are the targets)
            with torch.no_grad():
                teacher_enhanced_image, teacher_A = teacher_net(img_lowlight)

            # Get student outputs
            student_enhanced_image, student_A = student_net(img_lowlight)
            
            # Original losses for student
            Loss_TV = 1600 * L_TV(student_A)
            loss_spa = torch.mean(L_spa(student_enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(student_enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(student_enhanced_image, 0.6))

            # Knowledge distillation loss
            distillation_loss = knowledge_distillation_loss(
                (student_enhanced_image, student_A),
                (teacher_enhanced_image, teacher_A),
                img_lowlight
            )
            
            # Combined loss
            loss = config.alpha * distillation_loss + (1 - config.alpha) * (Loss_TV + loss_spa + loss_col + loss_exp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_net.parameters(), config.grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if ((iteration+1) % config.display_iter) == 0:
                print(f'Epoch {epoch+1}, Iteration {iteration+1}')
                print("Total Loss:", loss.item())
                print("Distillation Loss:", distillation_loss.item())
                print("Original Loss:", (Loss_TV + loss_spa + loss_col + loss_exp).item())
                
        # Save model after each epoch
        if not os.path.exists(config.snapshots_folder):
            os.mkdir(config.snapshots_folder)
            
        avg_epoch_loss = epoch_loss / num_batches
        print(f'Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss}')
        torch.save(student_net.state_dict(), 
                  config.snapshots_folder + f"Student_Epoch{epoch+1}.pth") 		

    # Save final model
    torch.save(student_net.state_dict(), config.snapshots_folder + "Student_Final.pth")


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
    parser.add_argument('--temperature', type=float, default=4.0, help='Temperature for soft targets')

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)