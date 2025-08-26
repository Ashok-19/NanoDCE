import torch
import torch.nn as nn
import os
import argparse
import time
import models.zerodce as zerodce
import models.model_student as model_student
import numpy as np
import cv2

class OptimizedWebcamProcessor:
    def __init__(self, model_path, scale_factor, model_type, camera_index=0):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            print("CUDA not available")
            return None
        
        # Initialize model
        if model_type == 'student':
            self.model_net = model_student.enhance_net_nopool_student(scale_factor).to(self.device)
        else:
            self.model_net = zerodce.enhance_net_nopool().to(self.device)
        
        self.model_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model_net.eval()
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")
        
        # Get properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.webcam_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.scale_factor = scale_factor
        
        # Calculate processing dimensions
        self.proc_h = (self.height // scale_factor) * scale_factor
        self.proc_w = (self.width // scale_factor) * scale_factor
        
        # Pre-allocate tensors to avoid memory allocation in loop
        self.input_tensor = torch.zeros(1, 3, self.proc_h, self.proc_w, device=self.device)
        
        # Create output buffer
        self.output_buffer = np.zeros((self.proc_h, self.proc_w, 3), dtype=np.uint8)
        
        print(f"Webcam: {self.width}x{self.height} @ {self.webcam_fps:.1f} FPS")
        print(f"Processing: {self.proc_w}x{self.proc_h}")
    
    def process_frame_optimized(self, frame):
        
        # Resize once (if needed)
        if self.proc_h != self.height or self.proc_w != self.width:
            frame = cv2.resize(frame, (self.proc_w, self.proc_h))
        
        # Convert BGR to RGB and normalize in one step
        frame_rgb = frame[:, :, ::-1]  
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        
        # Copy to pre-allocated tensor (avoids memory allocation)
        self.input_tensor[0].copy_(torch.from_numpy(frame_normalized).permute(2, 0, 1).to(self.device))
        
        # Process frame (GPU only)
        with torch.no_grad():
            enhanced_tensor, _ = self.model_net(self.input_tensor)
        
        # Convert back to numpy in one step
        enhanced_np = enhanced_tensor[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        
        # Convert RGB to BGR
        enhanced_bgr = enhanced_np[:, :, ::-1]  # RGB to BGR using slicing
        
        return enhanced_bgr
    
    def run(self):
        frame_count = 0
        fps_counter_start = time.time()
        frame_counter = 0
        current_fps = 0
        fullscreen = False
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_count += 1
                frame_counter += 1
                
                # Calculate FPS
                if frame_counter >= 30:
                    elapsed_time = time.time() - fps_counter_start
                    current_fps = frame_counter / elapsed_time
                    fps_counter_start = time.time()
                    frame_counter = 0
                
                # Process frame
                start_process = time.time()
                processed_frame = self.process_frame_optimized(frame)
                process_time = time.time() - start_process
                
                # Combine frames side by side
                combined_frame = np.hstack((frame, processed_frame))
                
                # Add labels and FPS
                cv2.putText(combined_frame, 'Original', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined_frame, 'Enhanced', (frame.shape[1] + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display all FPS info
                cv2.putText(combined_frame, f"Webcam: {self.webcam_fps:.1f}", 
                           (combined_frame.shape[1]//2 - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                
                cv2.putText(combined_frame, f"Model: {process_time*1000:.1f}ms", 
                           (combined_frame.shape[1]//2 - 50, combined_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Display frame
                if fullscreen:
                    cv2.namedWindow('Live Light Enhancement', cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty('Live Light Enhancement', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
                cv2.imshow('Live Light Enhancement', combined_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
                elif key == ord('f'):
                    fullscreen = not fullscreen
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"Processing stopped. Total frames: {frame_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='snapshots_Student_KD/Student_Final.pth')
    parser.add_argument('--model_type', type=str, default='student', choices=['student', 'teacher'])
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--camera_index', type=int, default=0)
    
    args = parser.parse_args()
    
    processor = OptimizedWebcamProcessor(
        args.model_path, args.scale_factor, args.model_type, args.camera_index
    )
    processor.run()