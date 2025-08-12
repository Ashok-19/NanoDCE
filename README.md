
# ZeroDCE_Extension-KD

This project is about using Knowledge Distillation method to compress the already small low light enhancement model [ZeroDCE++](https://github.com/Li-Chongyi/Zero-DCE_extension.git) to even a smaller version.

        Original Model - 10561 params

        Proposed Model - 565 params

## Knowledge Distillation in ZeroDCE++ Implementation  

### Fixed Teacher Model  
- Teacher network loaded from pre-trained weights and set to `eval()` mode  
- **Completely frozen**: No parameter updates via `torch.no_grad()` during teacher inference  
- Provides stable "soft targets" (enhanced images and illumination maps) as consistent learning references  
- *Critical implication*: Prevents moving target problem; student learns from fixed knowledge source  

### Feature Matching Mechanism  
- **Targeted layers**: Intermediate features extracted from `e_conv2`, `e_conv4`, and `e_conv6` layers  
- **Dynamic alignment**:  
  - Spatial dimensions matched via adaptive pooling  
  - Channel mismatches resolved through spatial averaging (`torch.mean(dim=1)`)  
- **Hierarchical weighting**: Loss contributions scaled by layer depth (0.2 for early, 0.3 mid, 0.5 deep layers)  
- *Core advantage*: Transfers teacher's internal representation strategy, not just final outputs  

### Combined Distillation Strategy  
Total loss = `α × output_distillation + β × feature_matching + (1-α-β) × task_losses`  
- Output distillation aligns student/teacher enhanced images and illumination maps  
- Feature matching ensures student replicates teacher's feature hierarchy  
- Task losses (color, exposure, smoothness) maintain domain-specific constraints  
- Default weights (α=0.7, β=0.2) prioritize output fidelity while preserving structural knowledge transfer

## Changes

Only Number of filter channels were changed -> from 32 to 4

The architecture of student model is kept the same as original 

        Original model --->

                    7 Conv Layers (Depthwise + Pointwise = 1 Conv layer)
                    8 iterations of LE curve

        Proposed model --->

                    7 Conv Layers (Depthwise + Pointwise = 1 Conv layer)
                    8 iterations of LE curve

The model architecture is not changed to preserve the Deep Neural architecture of the original model.

Refer [model.py](https://github.com/Ashok-19/ZeroDCE_extension-KD/blob/a86b2544082d32ca4af07b84b614cf8c721e2db3/model.py) and [model_student.py](https://github.com/Ashok-19/ZeroDCE_extension-KD/blob/a86b2544082d32ca4af07b84b614cf8c721e2db3/model_student.py) for Changes

## Prerequisites

* Pytorch - with cuda 
* opencv
* PIL
* numpy

## Training

Instead of using the same 2002 samples provided in the original model's [training data](https://github.com/Li-Chongyi/Zero-DCE_extension/tree/09f202b690f82da939b8e6ec8535960ae97ad8bd/Zero-DCE%2B%2B/data), an additional amount of another 2231 samples were added to original training data from datasets such as LIME, DICM, LOL-v2 and LoLI-street to avoid overfitting.


Before training create folder structure like below

                /data-
                   |-train_data
                   |-test_data


To train the student model, use

        python train_student_kd.py


## Testing

To test the student model, use

        python low_test.py --input_path=<path_to_image_folder/image/video> 


## Model size comparison

Refer [Param_check.ipynb](https://github.com/Ashok-19/ZeroDCE_extension-KD/blob/887e5b136b40b225530b755c9d447ee056435ebc/Param_check.ipynb)

## Results

Results for both test images and videos were uploaded in this [GDrive](https://drive.google.com/drive/folders/1-NzPEyCqdU4PwIbRDre48vN4SttSlAfv?usp=sharing)




