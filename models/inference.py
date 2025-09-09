import torch
import torch.nn as nn
import time
import numpy as np
import model_student as model_student
import cv2
import os
import model as zerodce_ext
def benchmark_model_inference(model_net, device, input_size=(1, 3, 512, 512), num_warmup=20, num_iterations=100):

    model_net.eval()
    
    dummy_input = torch.randn(input_size).to(device)
    
    print(f"Warming up for {num_warmup} iterations...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_net(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    print(f"Running benchmark for {num_iterations} iterations...")
    
    with torch.no_grad():
        for i in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model_net(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  
  
            if (i + 1) % 20 == 0:
                print(f"Completed {i + 1}/{num_iterations} iterations")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000.0 / avg_time
    
    return avg_time, fps, std_time


def count_model_parameters(model_net):
    total_params = sum(p.numel() for p in model_net.parameters())
    trainable_params = sum(p.numel() for p in model_net.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_size_mb(model_net):
    param_size = 0
    for param in model_net.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model_net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb

def benchmark_models_comprehensive(teacher_model_path, student_model_path, scale_factor=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Benchmarks require GPU.")
        return
    
    print(f"Using device: {device}")
    print("=" * 80)
    print("COMPREHENSIVE MODEL BENCHMARK")
    print("=" * 80)
    
    test_sizes = [
        (1, 3, 256, 256),
        (1, 3, 512, 512),
        (1, 3, 1024, 1024),
        (1,3,3840,2160)
    ]
    
    results = {}
    
    print("\n" + "-" * 40)
    print("BENCHMARKING TEACHER MODEL")
    print("-" * 40)

    
    teacher_net = zerodce_ext.enhance_net_nopool(scale_factor).to(device)
    teacher_net.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher_net.eval()
    
    teacher_params, _ = count_model_parameters(teacher_net)
    teacher_size = get_model_size_mb(teacher_net)
    
    print(f"Teacher Model Parameters: {teacher_params:,}")
    print(f"Teacher Model Size: {teacher_size:.2f} MB")
    
    teacher_results = {}
    for input_size in test_sizes:
        print(f"\nTesting input size: {input_size}")
        avg_time, fps, std_time = benchmark_model_inference(
            teacher_net, device, input_size, num_warmup=20, num_iterations=100
        )
        teacher_results[input_size] = {
            'avg_time': avg_time,
            'fps': fps,
            'std_time': std_time
        }
        print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")
    
    results['teacher'] = {
        'model': teacher_net,
        'params': teacher_params,
        'size': teacher_size,
        'results': teacher_results
    }
    
    print("\n" + "-" * 40)
    print("BENCHMARKING STUDENT MODEL")
    print("-" * 40)
    
    student_net = model_student.enhance_net_nopool_student(scale_factor).to(device)
    student_net.load_state_dict(torch.load(student_model_path, map_location=device))
    student_net.eval()
    
    student_params, _ = count_model_parameters(student_net)
    student_size = get_model_size_mb(student_net)
    
    print(f"Student Model Parameters: {student_params:,}")
    print(f"Student Model Size: {student_size:.2f} MB")
    
    student_results = {}
    for input_size in test_sizes:
        print(f"\nTesting input size: {input_size}")
        avg_time, fps, std_time = benchmark_model_inference(
            student_net, device, input_size, num_warmup=20, num_iterations=100
        )
        student_results[input_size] = {
            'avg_time': avg_time,
            'fps': fps,
            'std_time': std_time
        }
        print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  FPS: {fps:.2f}")
    
    results['student'] = {
        'model': student_net,
        'params': student_params,
        'size': student_size,
        'results': student_results
    }
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"{'Input Size':<15} {'Teacher Time (ms)':<20} {'Student Time (ms)':<20} {'Speedup':<10} {'Teacher FPS':<15} {'Student FPS':<15} {'FPS Gain':<10}")
    print("-" * 120)
    
    for input_size in test_sizes:
        teacher_time = results['teacher']['results'][input_size]['avg_time']
        student_time = results['student']['results'][input_size]['avg_time']
        teacher_fps = results['teacher']['results'][input_size]['fps']
        student_fps = results['student']['results'][input_size]['fps']
        
        speedup = teacher_time / student_time
        fps_gain = student_fps / teacher_fps
        
        print(f"{str(input_size[2:])+'px':<15} {teacher_time:<20.2f} {student_time:<20.2f} "
              f"{speedup:<10.2f}x {teacher_fps:<15.1f} {student_fps:<15.1f} {fps_gain:<10.2f}x")
    
    print("\n" + "-" * 50)
    print("MODEL SIZE COMPARISON")
    print("-" * 50)
    param_reduction = (results['teacher']['params'] - results['student']['params']) / results['teacher']['params'] * 100
    size_reduction = (results['teacher']['size'] - results['student']['size']) / results['teacher']['size'] * 100
    
    print(f"Teacher Parameters: {results['teacher']['params']:,}")
    print(f"Student Parameters: {results['student']['params']:,}")
    print(f"Parameter Reduction: {param_reduction:.1f}% ({results['teacher']['params']/results['student']['params']:.1f}x smaller)")
    print(f"Teacher Size: {results['teacher']['size']:.2f} MB")
    print(f"Student Size: {results['student']['size']:.2f} MB")
    print(f"Size Reduction: {size_reduction:.1f}% ({results['teacher']['size']/results['student']['size']:.1f}x smaller)")
 

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive benchmark of teacher and student models')
    parser.add_argument('--teacher_path', type=str, default='snapshots_Zero_DCE++/Epoch99.pth',
                       help='Path to teacher model weights')
    parser.add_argument('--student_path', type=str, default='snapshots_student_enhanced_kd/Student_Epoch100.pth',
                       help='Path to student model weights')
    parser.add_argument('--scale_factor', type=int, default=1,
                       help='Scale factor for models')
    
    args = parser.parse_args()
    
    benchmark_models_comprehensive(
        teacher_model_path=args.teacher_path,
        student_model_path=args.student_path,
        scale_factor=args.scale_factor
    )