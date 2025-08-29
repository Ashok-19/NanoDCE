# test.py (Updated with BRISQUE)
import torch
import torch.nn as nn
import torchvision
# Explicitly import transforms
from torchvision import transforms 
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import glob
import time
import model  # Teacher model definition (zerodce++)
import model_student  # Student model definition
# --- Import OpenCV for BRISQUE ---
import cv2 
from brisque.brisque import BRISQUE
# --- Define transforms globally ---
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# --- NEW FUNCTION: Calculate BRISQUE Score ---
def calculate_brisque_score(image_tensor):
    """
    Calculates the BRISQUE score for a PyTorch image tensor.
    Lower BRISQUE scores generally indicate better perceived image quality.
    Input: [B, C, H, W] tensor (values 0-1)
    Output: BRISQUE score (float) or None if calculation fails.
    """
    try:
        # Ensure the tensor is on the CPU for OpenCV processing
        image_tensor_cpu = image_tensor.detach().cpu()

        # Handle batch dimension: assume batch size 1 or squeeze if possible
        if image_tensor_cpu.shape[0] == 1:
            image_tensor_cpu = image_tensor_cpu.squeeze(0) # [C, H, W]
        elif image_tensor_cpu.dim() == 4:
             print("Warning: BRISQUE calculation expects batch size 1. Using first image.")
             image_tensor_cpu = image_tensor_cpu[0] # Take first image [C, H, W]

        # Convert from [C, H, W] tensor (0-1) to HWC NumPy array (0-255, uint8)
        # Permute channels: [C, H, W] -> [H, W, C]
        image_np = image_tensor_cpu.permute(1, 2, 0).numpy()
        ndarray = np.asarray(image_np)
        obj = BRISQUE(url=False)
        score = obj.score(img=ndarray)
        

        return score

    except Exception as e:
        print(f"Error calculating BRISQUE score: {e}")
        # Return None to indicate failure
        return None
# --- END NEW FUNCTION ---

def add_text_to_image(image_tensor, text_lines, font_size=20):
    """
    Adds multiple lines of text to a PyTorch tensor image.
    """
    # Convert tensor to PIL Image using the globally defined transform
    # Ensure tensor is on CPU for PIL conversion
    image_tensor_cpu = image_tensor.cpu() 
    image_pil = to_pil(image_tensor_cpu.squeeze(0)) # Remove batch dim
    
    # Prepare drawing context
    draw = ImageDraw.Draw(image_pil)
    
    # Try to load a default font, fallback to default if not found
    try:
        # Adjust path or font name as needed for your system
        font = ImageFont.truetype("DejaVuSans.ttf", font_size) 
    except OSError: # Catch specific error for file not found
        try:
            # Try another common one
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            # Fallback to default font if specific fonts are not found
            print("Warning: Specific font not found, using default.")
            font = ImageFont.load_default()
        
    text = "\n".join(text_lines)
    
    # Position text (e.g., top-left corner with a small margin)
    x, y = 10, 10
    
    # Draw text (white text with a slight black outline for better visibility)
    draw.text((x-1, y-1), text, font=font, fill="black")
    draw.text((x+1, y-1), text, font=font, fill="black")
    draw.text((x-1, y+1), text, font=font, fill="black")
    draw.text((x+1, y+1), text, font=font, fill="black")
    # Draw main text
    draw.text((x, y), text, font=font, fill="white")

    # Convert back to tensor using the globally defined transform
    # Ensure the PIL image is in RGB mode if the tensor was 3-channel
    if image_pil.mode != 'RGB' and image_tensor_cpu.shape[1] == 3:
        image_pil = image_pil.convert('RGB')
    image_tensor_result = to_tensor(image_pil).unsqueeze(0).to(image_tensor.device) # Re-add batch dim and move to original device

    return image_tensor_result


def lowlight(image_path, teacher_net, student_net, scale_factor):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_lowlight = Image.open(image_path)
    if data_lowlight.mode != 'RGB':
        data_lowlight = data_lowlight.convert('RGB')
        
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    
    # Ensure dimensions are divisible by scale_factor
    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0) # Add batch dimension

    # --- Teacher Inference ---
    teacher_net.eval()
    with torch.no_grad():
        start_teacher = time.time()
        teacher_output = teacher_net(data_lowlight)
        if isinstance(teacher_output, tuple):
            enhanced_teacher = teacher_output[0] # Assume first is image
        else:
            enhanced_teacher = teacher_output # Assume single output is image
        end_time_teacher = time.time() - start_teacher
        
        # --- Estimate BRISQUE score for teacher output ---
        brisque_teacher = calculate_brisque_score(enhanced_teacher)

    # --- Student Inference ---
    student_net.eval()
    with torch.no_grad():
        start_student = time.time()
        student_output = student_net(data_lowlight)
        if isinstance(student_output, tuple):
            enhanced_student = student_output[0] # Assume first is image
        else:
            enhanced_student = student_output # Assume single output is image
        end_time_student = time.time() - start_student
        
        # --- Estimate BRISQUE score for student output ---
        brisque_student = calculate_brisque_score(enhanced_student)

    # --- Prepare Output Paths ---
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    base_result_dir = image_path.replace('test_data', 'result_comparison')
    base_result_dir = os.path.dirname(base_result_dir) 
    
    teacher_result_path = os.path.join(base_result_dir, f"{base_name}_teacher.png")
    student_result_path = os.path.join(base_result_dir, f"{base_name}_student.png")

    os.makedirs(base_result_dir, exist_ok=True)

    # --- Annotate and Save Teacher Result ---
    # --- Updated text lines for BRISQUE ---
    teacher_text_lines = [
        f"Model: Teacher (ZeroDCE++)",
        f"Inference Time: {end_time_teacher:.4f}s",
        f"Estimated BRISQUE Score: {brisque_teacher:.2f}" if brisque_teacher is not None else "BRISQUE Score: Calculation Failed"
    ]
    try:
        annotated_teacher = add_text_to_image(enhanced_teacher, teacher_text_lines)
        torchvision.utils.save_image(annotated_teacher, teacher_result_path)
        print(f"Saved Teacher result: {teacher_result_path}")
    except Exception as e:
        print(f"Error saving Teacher result {teacher_result_path}: {e}")
        try:
            torchvision.utils.save_image(enhanced_teacher, teacher_result_path)
            print(f"Fallback: Saved unannotated Teacher result.")
        except:
            print("Failed to save Teacher result.")

    # --- Annotate and Save Student Result ---
    # --- Updated text lines for BRISQUE ---
    student_text_lines = [
        f"Model: Student (KD)",
        f"Inference Time: {end_time_student:.4f}s",
        f"Estimated BRISQUE Score: {brisque_student:.2f}" if brisque_student is not None else "BRISQUE Score: Calculation Failed"
    ]
    try:
        annotated_student = add_text_to_image(enhanced_student, student_text_lines)
        torchvision.utils.save_image(annotated_student, student_result_path)
        print(f"Saved Student result: {student_result_path}")
    except Exception as e:
        print(f"Error saving Student result {student_result_path}: {e}")
        try:
            torchvision.utils.save_image(enhanced_student, student_result_path)
            print(f"Fallback: Saved unannotated Student result.")
        except:
             print("Failed to save Student result.")

    # --- Return BRISQUE scores instead of Laplacian variance ---
    return end_time_teacher, end_time_student, brisque_teacher, brisque_student 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale_factor', type=int, default=1, help='Scale factor for the models')
    parser.add_argument('--teacher_model_path', type=str, default='snapshots_Zero_DCE++/Epoch-99_pre.pth', help='Path to the teacher model weights')
    parser.add_argument('--student_model_path', type=str, default='snapshots_Student_KD_both/Student_Epoch3.pth', help='Path to the student model weights')
    parser.add_argument('--test_data',type=str, default='data/test_data/',help='Path to Test data')
    args = parser.parse_args()

    scale_factor = args.scale_factor
    teacher_model_path = args.teacher_model_path
    student_model_path = args.student_model_path

    if not os.path.isfile(teacher_model_path):
        print(f"Error: Teacher model file not found at {teacher_model_path}")
        exit()
    if not os.path.isfile(student_model_path):
        print(f"Error: Student model file not found at {student_model_path}")
        exit()

    print("Loading Teacher Model...")
    # --- Corrected Teacher Model Instantiation ---
    # Use zerodce.enhance_net_nopool() as per your training setup
    #teacher_net = zerodce.enhance_net_nopool().cuda() 
    teacher_net = model.enhance_net_nopool(scale_factor).cuda() # Commented out
    teacher_net.load_state_dict(torch.load(teacher_model_path))
    print("Teacher Model loaded.")

    print("Loading Student Model...")
    # --- Corrected Student Model Instantiation ---
    # Use the 4-conv student model class name you defined
    student_net = model_student.enhance_net_nopool_student(scale_factor).cuda() 
    #student_net = model.enhance_net_nopool(scale_factor).cuda() # Commented out
    student_net.load_state_dict(torch.load(student_model_path))
    print("Student Model loaded.")

    filePath = args.test_data
    sum_time_teacher = 0
    sum_time_student = 0
    # --- Updated lists to store BRISQUE scores ---
    all_brisque_teacher = [] 
    all_brisque_student = []

    if not os.path.exists(filePath):
        print(f"Error: Test data path '{filePath}' does not exist.")
        exit()

    items_in_base = os.listdir(filePath) 
    subfolders = [item for item in items_in_base if os.path.isdir(os.path.join(filePath, item))]
    base_files = [item for item in items_in_base if os.path.isfile(os.path.join(filePath, item)) and item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]

    test_image_paths = []
    for subfolder_name in subfolders:
        subfolder_path = os.path.join(filePath, subfolder_name)
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']:
             test_image_paths.extend(glob.glob(os.path.join(subfolder_path, ext)))
    for file_name in base_files:
        test_image_paths.append(os.path.join(filePath, file_name))

    print(f"Found {len(test_image_paths)} image(s) to process.")

    for image_path in test_image_paths:
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            print(f"\nProcessing: {image_path}")
            try:
                # --- Updated variable names to reflect BRISQUE ---
                t_time, s_time, b_t, b_s = lowlight(image_path, teacher_net, student_net, scale_factor)
                sum_time_teacher += t_time
                sum_time_student += s_time
                # --- Append BRISQUE scores, handling potential None values ---
                if b_t is not None:
                    all_brisque_teacher.append(b_t)
                else:
                     print(f"Warning: BRISQUE calculation failed for Teacher on {image_path}")
                if b_s is not None:
                    all_brisque_student.append(b_s)
                else:
                     print(f"Warning: BRISQUE calculation failed for Student on {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Skipping non-image file: {image_path}")

    # --- Print Summary using BRISQUE ---
    # --- Updated variable names and messages ---
    num_images_processed = len(all_brisque_teacher) # Should match len(all_brisque_student) for successful ones
    num_images_attempted = len(test_image_paths) # Total images attempted
    if num_images_processed > 0:
        avg_time_teacher = sum_time_teacher / num_images_attempted # Average time over all attempted images
        avg_time_student = sum_time_student / num_images_attempted
        avg_brisque_teacher = np.mean(all_brisque_teacher)
        avg_brisque_student = np.mean(all_brisque_student)
        std_brisque_teacher = np.std(all_brisque_teacher)
        std_brisque_student = np.std(all_brisque_student)

        print("\n--- Summary ---")
        print(f"Total Images Attempted: {num_images_attempted}")
        print(f"Total Images Successfully Processed (BRISQUE calculated): {num_images_processed}")
        print(f"Average Teacher Inference Time: {avg_time_teacher:.4f}s")
        print(f"Average Student Inference Time: {avg_time_student:.4f}s")
        print(f"Average Estimated BRISQUE Score:")
        print(f"  Teacher: {avg_brisque_teacher:.2f} (+/- {std_brisque_teacher:.2f})")
        print(f"  Student: {avg_brisque_student:.2f} (+/- {std_brisque_student:.2f})")

        # --- Simple comparison based on BRISQUE ---
        # Note: Lower BRISQUE score indicates better perceived quality.
        if avg_brisque_student < avg_brisque_teacher:
             diff_percent = ((avg_brisque_teacher - avg_brisque_student) / avg_brisque_teacher) * 100
             print(f"\nStudent model produced outputs estimated to have {diff_percent:.2f}% better quality (lower BRISQUE) on average.")
        elif avg_brisque_student > avg_brisque_teacher:
             diff_percent = ((avg_brisque_student - avg_brisque_teacher) / avg_brisque_teacher) * 100
             print(f"\nTeacher model produced outputs estimated to have {diff_percent:.2f}% better quality (lower BRISQUE) on average.")
        else:
             print("\nAverage estimated BRISQUE score is the same for both models.")
    else:
        print("No images were successfully processed (BRISQUE calculation failed for all).")
