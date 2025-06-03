# CNN-Pawpularity-Score-Prediction
This project aims to help increase pet adoption rates by predicting the Pawpularity Score of pet photos based on their visual appeal and metadata. It uses a deep learning, convolutional neural network, model combining image data and tabular features to estimate how appealing a pet photo is to potential adopters.

### Problem Statement
Millions of pets struggle to find homes, and their chances often depend on the appeal of their online profile photos. PetFinder.my currently uses a basic "Cuteness Meter," which lacks sophistication. Our goal is to build a regression model that can accurately predict a pet’s Pawpularity Score using both the image and its descriptive metadata.

### Objective
Build a predictive model using computer vision and tabular data that outputs a continuous Pawpularity Score, helping PetFinder.my identify and promote the most appealing pet photos to boost adoption.

### Dataset Overview
- 📷 9,912 training images with corresponding tabular metadata
- 🐶 ~6,800 test images without scores (for leaderboard evaluation)
- 🧾 Metadata includes 13 binary features (e.g., "focus", "bright", "human", "eyes_visible")

### Data Preprocessing
- All images resized to (224, 224, 3) for consistency
- Data augmentation: rotation, flipping, blurring to improve generalization
- Image normalization and tabular feature scaling applied
- Custom RMSE function used for evaluation

### Model Architecture

### 🔷 Dual-Input Neural Network

### 📸 Image Pipeline
Pretrained EfficientNetB0 with include_top=False
Last 15 layers unfrozen for transfer learning
Output flattened from (7, 7, 1280) → 62720 vector

### 📊 Metadata Pipeline
13 binary features → Dense(32, activation='tanh') + Dropout(0.2)

### 🔁 Fusion + Final Layers
Concatenated image + metadata → 62752 vector
Dense(64) → Dropout(0.2) → Dense(128) → Output (1 neuron, linear)

### 🔧 Compilation
Loss: Mean Squared Error (MSE)
Optimizer: Adam (lr=0.0003)
Evaluation: Root Mean Squared Error (RMSE)

### Libraries Used
- tensorflow, keras, opencv-python
- numpy, pandas, matplotlib, seaborn
- scikit-learn

### Key Learnings
- Transfer learning significantly speeds up training and improves results with limited data.
- Combining image features with metadata leads to better predictive performance.
- Data augmentation and dropout layers are critical to reducing overfitting.
- EfficientNetB0 is a powerful yet lightweight model for production-grade applications.

### Future Work
- Deploy the model with Streamlit or Gradio for interactive use
- Add attention mechanisms or experiment with Vision Transformers
- Use K-Fold Cross-Validation for better generalization
- Implement inference pipeline for real-world use
