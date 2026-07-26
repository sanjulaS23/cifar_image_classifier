import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr
from model import CIFAR10_CNN  # Make sure your model architecture class is available

# 1. Classes in CIFAR-10
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 2. Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10_CNN(num_classes=10).to(device)

checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# 3. Image preprocessing (CIFAR-10 expects 32x32 images with specific normalization)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 4. Prediction function for Gradio
def predict(image):
    if image is None:
        return "Please upload an image."
    
    # Preprocess image
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)[0]
    
    # Create a dictionary of class confidences
    confidences = {classes[i]: float(probabilities[i]) for i in range(10)}
    return confidences

# 5. Build Gradio Interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Image Classifier",
    description="Upload an image (like an airplane, car, cat, dog, etc.) and see what the AI model predicts!"
)

if __name__ == "__main__":
    demo.launch(share=True)  # share=True will give you a public public link in Colab!