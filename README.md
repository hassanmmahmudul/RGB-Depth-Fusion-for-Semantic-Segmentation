# RGB-Depth Fusion for Semantic Segmentation

## 📌 Project Overview
This project implements an **RGB-Depth Fusion architecture** for **semantic segmentation** using a **Fully Convolutional Network (FCN)**. The architecture is based on a two-stream approach, where **RGB and Depth** modalities are processed separately before being fused for segmentation. 

## 📂 Dataset
- The dataset consists of **1100 images per modality (RGB, Depth)** from road scenes.
- It is split into:
  - **Training Set**: 600 images
  - **Validation Set**: 300 images
  - **Test Set**: 200 images
- Each image is resized to **256 × 256**.
- **Pixel-level annotations** for **19 semantic classes** are provided.

## 🚀 Model Architecture
The **Fully Convolutional Network (FCN)** is built as follows:

1. **Two parallel streams**:
   - **Pretrained ResNet50** (on ImageNet) is used as a feature extractor.
   - **One stream processes RGB images**, and the **other processes Depth images**.

2. **Convolutional layers**:
   - Two convolution layers per stream:
     - **128 filters**, kernel size **(3,3)**, stride **(1,1)**
     - **256 filters**, kernel size **(3,3)**, stride **(1,1)**
   - Followed by a **dropout layer (rate=0.2)**

3. **Fusion step**:
   - The outputs of both streams are **concatenated**.

4. **Upsampling using Transposed Convolution**:
   - **Kernel size (64,64)**, stride **(32,32)**

5. **Final layers**:
   - **Reshape layer** to match segmentation output
   - **Softmax activation** for pixel classification

## 🔧 Model Compilation & Training
- **Optimizer**: `SGD(learning_rate=0.008, decay=1e-6, momentum=0.9)`
- **Loss Function**: `categorical_crossentropy`
- **Metrics**: `accuracy`
- **Epochs**: `10`
- **Training Data**: Uses both training and validation datasets.

## 📊 Evaluation & Predictions
- The model is evaluated on the **test dataset**.
- Prints **Test Loss & Accuracy**.
- Predicts **5 randomly selected test images** and visualizes:
  - **RGB Image**
  - **Depth Image**
  - **Ground Truth (GT) Segmentation**
  - **Predicted Segmentation**

## 💡 Project demonstrates: 
- **Fusion of RGB and Depth information** for enhanced segmentation.
- **Usage of pretrained networks** (ResNet50) for feature extraction.
- **Fully Convolutional Networks (FCN)** for semantic segmentation.
- **Evaluation of different modalities (RGB vs. Depth vs. Fusion)**.

## 🔗 References
- [FCN for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [ResNet50](https://arxiv.org/abs/1512.03385)
