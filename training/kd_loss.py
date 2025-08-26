from torchmetrics.functional import structural_similarity_index_measure
import torch
import torch.functional
import torch.nn as nn


def feature_matching_loss(student_features, teacher_features):
    mse_loss = nn.MSELoss()
    # Initialize feature_loss as a PyTorch tensor on the same device as inputs
    device = next(iter(student_features.values())).device if student_features else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feature_loss = torch.tensor(0.0, device=device)

    # Define weights for different student layers being matched
    student_layer_weights = {
        'e_conv1': 0.2,
        'e_conv2': 0.3,
        'e_conv3': 0.5
    }

    computed_any_loss = False
    # Iterate through student features
    for student_feature_key, student_feat in student_features.items():
        # Parse the student feature key
        if student_feature_key.startswith("student_") and "_matches_teacher_" in student_feature_key:
            parts = student_feature_key.split("_matches_teacher_")
            if len(parts) == 2:
                student_layer_part = parts[0].replace("student_", "", 1)
                teacher_layer_part = parts[1]

                # Construct the expected teacher feature key name
                expected_teacher_feature_key = f"{teacher_layer_part}_features"

                # Check if the corresponding teacher feature exists
                if expected_teacher_feature_key in teacher_features:
                    teacher_feat = teacher_features[expected_teacher_feature_key].detach()

                    # Handle spatial dimension mismatch
                    if student_feat.shape[2:] != teacher_feat.shape[2:]:
                        teacher_feat = nn.functional.adaptive_avg_pool2d(teacher_feat, student_feat.shape[2:])

                    # Handle channel dimension mismatch
                    if student_feat.shape[1] != teacher_feat.shape[1]:
                        # Reduce both to spatial information only (GAP)
                        student_spatial = torch.mean(student_feat, dim=1, keepdim=True)
                        teacher_spatial = torch.mean(teacher_feat, dim=1, keepdim=True)
                        if student_spatial.shape[2:] != teacher_spatial.shape[2:]:
                            teacher_spatial = nn.functional.adaptive_avg_pool2d(teacher_spatial, student_spatial.shape[2:])
                        student_feat = student_spatial
                        teacher_feat = teacher_spatial

                    # Get the weight for this specific student layer match
                    weight = student_layer_weights.get(student_layer_part, 0.25)

                    # Accumulate weighted MSE loss
                    feature_loss = feature_loss + weight * mse_loss(student_feat, teacher_feat)
                    computed_any_loss = True
                else:
                    print(f"Warning: Expected teacher feature '{expected_teacher_feature_key}' not found.")
            else:
                 print(f"Warning: Could not parse student feature key '{student_feature_key}'.")
        else:
             pass

    if not computed_any_loss:
         print("Warning: No feature pairs were matched.")

    return feature_loss

def knowledge_distillation_loss(student_output, teacher_output, vgg_perceptual_loss=None, temperature=4.0):
    mse_loss = nn.MSELoss()
    # Base pixel loss
    pixel_loss = mse_loss(student_output[0], teacher_output[0])
    
    # VGG perceptual loss (weight 0.1 as suggested in research)
    if vgg_perceptual_loss is not None:
        vgg_loss = vgg_perceptual_loss(student_output[0], teacher_output[0]) * 0.1
    else:
        vgg_loss = 0.0
    
    # SSIM loss (weight 0.05 as suggested in research) - using PyTorch's built-in
    ssim_loss = (1 - structural_similarity_index_measure(student_output[0], teacher_output[0])) * 0.05
    
    # Combined loss components
    return pixel_loss + vgg_loss + ssim_loss