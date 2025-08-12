import torch
import torch.nn as nn
import torchvision
import os
import argparse
import time
import model
import model_student
import numpy as np
import cv2

def setup_model(model_path, scale_factor, model_type):
    
    # Ensure CUDA is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Cuda not available")
        return None, None
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # Initialize model
    if model_type == 'student':
        model_net = model_student.enhance_net_nopool_student(scale_factor).to(device)
    else:
        model_net = model.enhance_net_nopool(scale_factor).to(device)
    
    # Load model weights
    model_net.load_state_dict(torch.load(model_path, map_location=device))
    model_net.eval()
    
    return model_net, device

def process_frame(frame, model_net, device, scale_factor):
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Calculate resized dimensions (must be divisible by scale_factor)
    h = (height // scale_factor) * scale_factor
    w = (width // scale_factor) * scale_factor
    
    # Resize frame if necessary
    if h != height or w != width:
        frame_resized = cv2.resize(frame, (w, h))
    else:
        frame_resized = frame
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize and convert to tensor
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    frame_tensor = torch.from_numpy(frame_normalized).float()
    frame_tensor = frame_tensor.permute(2, 0, 1).to(device).unsqueeze(0)
    
    # Process frame
    with torch.no_grad():
        enhanced_frame, _ = model_net(frame_tensor)
    
    # Convert back to numpy
    enhanced_frame = enhanced_frame.squeeze(0).permute(1, 2, 0)
    enhanced_frame = torch.clamp(enhanced_frame, 0, 1)
    enhanced_frame_np = (enhanced_frame.cpu().numpy() * 255).astype(np.uint8)
    
    # Convert RGB back to BGR
    enhanced_frame_bgr = cv2.cvtColor(enhanced_frame_np, cv2.COLOR_RGB2BGR)
    
    return enhanced_frame_bgr

def live_webcam_processing(model_path, scale_factor, model_type, camera_index=0):
    
    print("Setting up model...")
    model_net, device = setup_model(model_path, scale_factor, model_type)
    
    if model_net is None:
        return
    
    print("Initializing webcam...")
    # Initialize webcam
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Webcam initialized: {width}x{height} @ {fps} FPS")
    print("Press 'q' or 'ESC' to quit")
    print("Press 'f' to toggle fullscreen mode")
    
    frame_count = 0
    start_time = time.time()
    fps_counter_start = time.time()
    frame_counter = 0
    current_fps = 0
    fullscreen = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            frame_count += 1
            frame_counter += 1
            
            # Calculate FPS every 10 frames for smoother display
            if frame_counter >= 10:
                elapsed_time = time.time() - fps_counter_start
                current_fps = frame_counter / elapsed_time
                fps_counter_start = time.time()
                frame_counter = 0
            
            # Process frame to get enhanced version
            processed_frame = process_frame(frame, model_net, device, scale_factor)
            
            # Resize frames to same height for side-by-side display
            orig_h, orig_w = frame.shape[:2]
            proc_h, proc_w = processed_frame.shape[:2]
            
            # Use the smaller height for both frames
            target_height = min(orig_h, proc_h)
            
            # Resize both frames to same height
            if orig_h != target_height:
                new_width = int(orig_w * (target_height / orig_h))
                frame_resized = cv2.resize(frame, (new_width, target_height))
            else:
                frame_resized = frame
                
            if proc_h != target_height:
                new_width = int(proc_w * (target_height / proc_h))
                processed_resized = cv2.resize(processed_frame, (new_width, target_height))
            else:
                processed_resized = processed_frame
            
            # Combine frames side by side
            combined_frame = np.hstack((frame_resized, processed_resized))
            
            # Add labels with FPS
            cv2.putText(combined_frame, 'Original', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined_frame, 'Enhanced', (frame_resized.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            
            # Display model info
            model_text = f"Model: {model_type.upper()}"
            cv2.putText(combined_frame, model_text, (combined_frame.shape[1] - 200, combined_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Display frame counter
            frame_text = f"Frame: {frame_count}"
            cv2.putText(combined_frame, frame_text, (combined_frame.shape[1] // 2 - 50, combined_frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Display frame
            if fullscreen:
                cv2.namedWindow('Live Light Enhancement - Side by Side', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Live Light Enhancement - Side by Side', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            cv2.imshow('Live Light Enhancement - Side by Side', combined_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break
            elif key == ord('f'):  # 'f' to toggle fullscreen
                fullscreen = not fullscreen
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam processing stopped")
        print(f"Total frames processed: {frame_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='snapshots_Student_KD_both/Student_Final.pth', 
                       help='Path to model weights')
    parser.add_argument('--model_type', type=str, default='student', 
                       choices=['student', 'teacher'], help='Model type')
    parser.add_argument('--scale_factor', type=int, default=1, 
                       help='Scale factor (use 1 for real-time)')
    parser.add_argument('--camera_index', type=int, default=0, 
                       help='Camera index (0 for default webcam)')
    
    args = parser.parse_args()
    
    print("Starting live webcam processing...")
    print(f"Model: {args.model_type}")
    print(f"Model path: {args.model_path}")
    print(f"Scale factor: {args.scale_factor}")
    print("Display: Side-by-side original and enhanced with FPS")
    
    live_webcam_processing(
        model_path=args.model_path,
        scale_factor=args.scale_factor,
        model_type=args.model_type,
        camera_index=args.camera_index
    )