
# ZeroDCE_Extension-KD

Introducing the smallest variant of ZeroDCE family - **TinyDCE** with only 161 parameters that achieves similar results of ZeroDCE and ZeroDCE++

This project is about using Knowledge Distillation method to compress the already small low light enhancement model [ZeroDCE++](https://github.com/Li-Chongyi/Zero-DCE_extension.git) to an even smaller version resulting in 98.5% (65.6x) smaller than ZeroDCE++ and 99.8% (493.3x) smaller than ZeroDCE while retaining similar results.

        ZeroDCE   - 79416 params
        ZeroDCE++ - 10561 params

        TinyDCE   - 161 params (Proposed)

## Knowledge Distillation Training Method- visit [here](https://github.com/Ashok-19/TinyDCE/blob/bbd0cb20cf0a780499f56cc79b709ebac495b40e/sample_output.txt)

## Architectural Changes
The student retains Zero-DCE++'s core (curve estimation via iterative LE maps) but compresses aggressively:

_Filter Channels_: Reduced from 32 to 4.
_Convolutional Layers_: From 7 (Zero-DCE++) to 3 (each Depthwise + Pointwise as one logical conv).
_Curve Iterations_: From 8 to 4, trading some dynamic range for efficiency.
_Implication_: Preserves U-Net-like skip connections but trims depth, relying on KD to recover lost capacity.

This yields ~161 params in the trained student model.



## Inference times

This script will provide an approximate inference results but not an accurate one.

        python inference.py 

#### Sample Output

Check the sample output [here](https://github.com/Ashok-19/TinyDCE/blob/bbd0cb20cf0a780499f56cc79b709ebac495b40e/sample_output.txt)

## Live Video Performance

As for the live video performance, the latency for each frame is calculated and displayed.

To check live video performance, run

                #for student model
        python live_webcam.py --model_path=snapshots_student_enhanced_kd/Student_Epoch100.pth  --enable_detection

                #for teacher model
        python live_webcam.py --model_type=teacher --model_path=snapshots_Zero_DCE++/Epoch99-pre.pth  --enable_detection

Note: YOLO detection is optional

* Performance (without using YOLO detection)
  
          Student  |  Teacher
          -------- | ---------
          6.1-7 ms | 35-45 ms

  Note: Live webcam performance may vary for each system, Here the system configs were,
  * CPU - Ryzen 7 4800H
  * GPU - RTX 3050
          

Refer [model.py](https://github.com/Ashok-19/TinyDCE/blob/c6e33399ecc731c8d72d6760062a3ca86d126113/models/model.py) and [model_student.py](https://github.com/Ashok-19/TinyDCE/blob/c6e33399ecc731c8d72d6760062a3ca86d126113/models/model_student.py) for Changes

## Prerequisites

* Pytorch - with cuda 
* opencv
* PIL
* numpy

## Training

Instead of using the same 2002 samples provided in the original model's [training data](https://github.com/Li-Chongyi/Zero-DCE_extension/tree/09f202b690f82da939b8e6ec8535960ae97ad8bd/Zero-DCE%2B%2B/data), an additional amount of another 4089 samples were added to original training data from various datasets such as LIME, DICM, LOL-v1, LOL-v2 ,LSRW (both huawei and nikon) and LoLI-street to avoid overfitting.


Before training create folder structure like below in the training directory.

                /data-
                   |-train_data
                   |-test_data


To train the student model, use

        python train_student_kd.py


## Testing

To test the student model, use

        python lowlight_test.py --test_data=<> --student_model_path=<> --teacher_model_path=<>


## Model size comparison

Refer [Param_check.ipynb]()

## FPGA Implementation
To be updated...

## Results

Results for both test images and videos were uploaded in this [GDrive]()


## Contact

Mail -> ashokraja1910@gmail.com

