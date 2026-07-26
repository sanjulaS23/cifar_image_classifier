# CIFAR-10 Image Classification with Custom CNN, Gradio, and PyTorch

An end-to-end deep learning project that implements a custom Convolutional Neural Network (CNN) from scratch using **PyTorch** to classify images from the **CIFAR-10** dataset. The project includes complete pipelines for data preprocessing, model training, evaluation, automated testing, and an interactive web application built with **Gradio** for real-time image classification.

---

# Project Overview & Features

- **Custom CNN Architecture:** Designed from scratch using convolutional layers, batch normalization, ReLU activation functions, max pooling, and dropout to improve performance and reduce overfitting.
- **Model Training & Evaluation:** Trained using data augmentation, cross-entropy loss, and the Adam optimizer, achieving reliable classification performance on the CIFAR-10 dataset.
- **Testing Script (`src/test.py`):** Automatically loads the best saved model, selects a random test image, and displays the true and predicted class labels.
- **Interactive Web Application (`src/app.py`):** Built with **Gradio**, allowing users to upload custom images and receive real-time predictions with confidence scores.

---

# Project Structure

```text
cifar_image_classifier/
│
├── checkpoints/
│   └── best_model.pth        # Trained model weights
│
├── data/                     # CIFAR-10 dataset (downloaded automatically)
│
├── src/
│   ├── model.py              # CNN model architecture
│   ├── train.py              # Model training and validation
│   ├── test.py               # Test the model on random images
│   └── app.py                # Gradio web application
│
├── .gitignore                # Git ignored files
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

# Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sanjulaS23/cifar_image_classifier.git
cd cifar_image_classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# How to Run

## 1. Test the Model (Command-Line Inference)

Run the following command to load the trained model and evaluate random test images:

```bash
python src/test.py
```

---

## 2. Launch the Gradio Web Application

Start the interactive interface:

```bash
python src/app.py
```

This will generate:

- A local URL (e.g., `http://127.0.0.1:7860`)
- A public `.gradio.live` URL for temporary online access

---

# Tech Stack & Libraries

- Python
- PyTorch
- Torchvision
- Gradio
- NumPy
- Matplotlib
- Git
- GitHub
