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
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import cv2
import zerodce

def process_image(image_path, model_path, scale_factor, model_type):
    # Ensure CUDA is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    data_lowlight = Image.open(image_path)
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    
    data_lowlight = torch.from_numpy(data_lowlight).float()
    
    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)  # Move to device
    
    # Initialize model
    if model_type == 'student':
        DCE_net = model_student.enhance_net_nopool_student(scale_factor).to(device)  # Move to device
    else:
        #DCE_net = model.enhance_net_nopool(scale_factor).to(device)  # Move to device
        DCE_net = zerodce.enhance_net_nopool().to(device)  # Move to device
    
    DCE_net.load_state_dict(torch.load(model_path, map_location=device))  # Load with device mapping
    DCE_net.eval()  # Set to evaluation mode
    
    start = time.time()
    enhanced_image, params_maps = DCE_net(data_lowlight)
    end_time = (time.time() - start)
    
    print(f"Image processing time: {end_time:.4f} seconds")
    
    # Save result
    if model_type == 'student':
        result_path = image_path.replace('test_data', 'result_student')
    else:
        result_path = image_path.replace('test_data', 'result_teacher')
    
    if not os.path.exists(result_path.replace('/' + result_path.split("/")[-1], '')):
        os.makedirs(result_path.replace('/' + result_path.split("/")[-1], ''))
    
    # Move tensor to CPU for saving
    torchvision.utils.save_image(enhanced_image.cpu(), result_path)
    return end_time

def process_video(video_path, model_path, scale_factor, model_type, output_fps=None):
    # Ensure CUDA is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Initialize model
    if model_type == 'student':
        DCE_net = model_student.enhance_net_nopool_student(scale_factor).to(device)  # Move to device
    else:
        DCE_net = zerodce.enhance_net_nopool().to(device)  # Move to device
    
    DCE_net.load_state_dict(torch.load(model_path, map_location=device))  # Load with device mapping
    DCE_net.eval()  # Set to evaluation mode
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if output_fps is None:
        output_fps = fps
    
    # Calculate resized dimensions (must be divisible by scale_factor)
    h = (height // scale_factor) * scale_factor
    w = (width // scale_factor) * scale_factor
    
    # Create output video path
    if model_type == 'student':
        output_path = video_path.replace('test_data', 'result_student').replace('.mp4', '_enhanced.mp4')
    else:
        output_path = video_path.replace('test_data', 'result_teacher').replace('.mp4', '_enhanced.mp4')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))
    
    print(f"Processing video: {video_path}")
    print(f"Input size: {width}x{height}, Output size: {w}x{h}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    frame_count = 0
    total_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to be divisible by scale_factor
        frame_resized = cv2.resize(frame_rgb, (w, h))
        
        # Normalize and convert to tensor
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_tensor = torch.from_numpy(frame_normalized).float()
        frame_tensor = frame_tensor.permute(2, 0, 1).to(device).unsqueeze(0)  # Move to device
        
        # Process frame
        start_time = time.time()
        enhanced_frame, _ = DCE_net(frame_tensor)
        frame_time = time.time() - start_time
        total_time += frame_time
        
        # Convert back to numpy and save
        enhanced_frame = enhanced_frame.squeeze(0).permute(1, 2, 0)
        enhanced_frame = torch.clamp(enhanced_frame, 0, 1)
        enhanced_frame_np = (enhanced_frame.cpu().numpy() * 255).astype(np.uint8)  # Move to CPU for conversion
        
        # Convert RGB back to BGR for OpenCV
        enhanced_frame_bgr = cv2.cvtColor(enhanced_frame_np, cv2.COLOR_RGB2BGR)
        out.write(enhanced_frame_bgr)
        
        if frame_count % 30 == 0:  # Print progress every 30 frames
            print(f"Processed frame {frame_count}/{total_frames}, Time: {frame_time:.4f}s")
    
    # Release everything
    cap.release()
    out.release()
    
    avg_time = total_time / frame_count if frame_count > 0 else 0
    print(f"Video processing completed!")
    print(f"Total frames: {frame_count}, Average time per frame: {avg_time:.4f}s")
    print(f"Output saved to: {output_path}")
    
    return total_time

def lowlight(input_path, model_path, scale_factor, model_type, process_type='image'):
    if process_type == 'image':
        return process_image(input_path, model_path, scale_factor, model_type)
    elif process_type == 'video':
        return process_video(input_path, model_path, scale_factor, model_type)
    else:
        raise ValueError("process_type must be 'image' or 'video'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='data/test_data/', help='Input path (image/video or directory)')
    parser.add_argument('--model_path', type=str, default='snapshots_Student_KD_both/Student_Epoch100.pth', help='Path to model weights')
    parser.add_argument('--model_type', type=str, default='student', choices=['student', 'teacher'], help='Model type')
    parser.add_argument('--scale_factor', type=int, default=1, help='Scale factor')
    parser.add_argument('--process_type', type=str, default='auto', choices=['image', 'video', 'auto'], help='Process type')
    parser.add_argument('--output_fps', type=int, default=None, help='Output FPS for video (default: same as input)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Please check your CUDA installation.")
        sys.exit(1)
    
    print(f"Using device: CUDA")
    
    with torch.no_grad():
        # Check if input is a file or directory
        if os.path.isfile(args.input_path):
            # Single file processing
            file_ext = os.path.splitext(args.input_path)[1].lower()
            
            if args.process_type == 'auto':
                if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    process_type = 'video'
                else:
                    process_type = 'image'
            else:
                process_type = args.process_type
            
            print(f"Processing single {process_type}: {args.input_path}")
            lowlight(args.input_path, args.model_path, args.scale_factor, args.model_type, process_type)
            
        else:
            # Directory processing
            filePath = args.input_path
            file_list = os.listdir(filePath)
            sum_time = 0
            
            for file_name in file_list:
                test_list = glob.glob(os.path.join(filePath, file_name, "*"))
                for item in test_list:
                    file_ext = os.path.splitext(item)[1].lower()
                    
                    if args.process_type == 'auto':
                        if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                            process_type = 'video'
                        elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                            process_type = 'image'
                        else:
                            continue  # Skip unsupported files
                    else:
                        process_type = args.process_type
                    
                    print(f"Processing {process_type}: {item}")
                    sum_time += lowlight(item, args.model_path, args.scale_factor, args.model_type, process_type)
            
            print(f"Total processing time: {sum_time:.4f} seconds")