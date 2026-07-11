# Dermān: Skin Disease Detection

An AI-powered web application that analyzes skin conditions from user-uploaded images. Designed with a premium, responsive user interface and powered by a custom-trained deep learning model, Dermān can detect and classify over 23 distinct dermatological conditions.

## 🚀 Features

- **Premium User Interface**: A modern, responsive, and accessible web interface built with vanilla HTML, CSS, and JS. Features glassmorphism effects, dynamic modals, and mobile-first design.
- **AI-Powered Analysis**: Instant analysis of skin images using a highly optimized deep learning pipeline.
- **Detailed Insights**: Provides predictions alongside confidence scores across 23+ disease categories.

## 🧠 Model Details

The core of the application utilizes a highly advanced **Ensemble Architecture**, running two state-of-the-art vision models simultaneously to cross-validate and average predictions for maximum clinical accuracy.

- **Architecture**: Ensemble (ConvNeXt-Base + EfficientNetV2-S)
- **Input Resolution**: 384x384 pixels
- **Classes**: 23 distinct skin condition categories (including Acne, Eczema, Melanoma, Psoriasis, etc.)
- **Performance**: 
  - **Test Accuracy**: ~75%+ (Clinical Grade for this dataset complexity)
- **Advanced Training Techniques Applied**:
  - **Dual-Model Inference**: Averages the probability distributions of ConvNeXt and EfficientNet for highly stable predictions.
  - **Regularization**: MixUp (alpha=0.4) + CutMix (alpha=1.0) applied to 80% of batches
  - **Classifier Head**: Custom 2-layer head (Linear -> GELU -> Linear)
  - **Optimization**: Differential Learning Rates (backbone 5e-5 / head 5e-4) with CosineAnnealingWarmRestarts
  - **Inference**: 7-augmentation Test-Time Augmentation (TTA)

## 🛠️ Tech Stack

- **Backend**: Flask (Python)
- **Machine Learning**: PyTorch / TIMM (EfficientNet)
- **Frontend**: HTML5, CSS3 (Custom Variables, Flexbox/Grid, Animations), JavaScript (Vanilla)

## 💻 Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Amaan-Parvaiz-Rather/Skin-Disease-Detection.git
   cd Skin-Disease-Detection
   ```

2. **Download the Models**:
   - Download the model `.pth` files from this [Google Drive Link](https://drive.google.com/drive/folders/10PnewspGtjBA7gERg7mtMnHflNpcQEp_?usp=sharing).
   - Place all the downloaded `.pth` files inside the `Checkpoints` directory in the root of the project.

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```

5. **Open in Browser**:
   Navigate to `http://127.0.1:5000`

---
*Disclaimer: Dermān provides informational analysis based on visual data. It does not replace professional dermatological consultation.*
