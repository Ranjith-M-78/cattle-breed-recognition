import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pickle

st.title("🐄 Cow Breed Prediction App")

# ✅ Load correct class names
with open("classes.pkl", "rb") as f:
    class_names = pickle.load(f)

# ✅ Load model
model = torch.load("model2.pkl", map_location='cpu', weights_only=False)
model.eval()

# ✅ SAME transform as training (NO flip)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

uploaded_file = st.file_uploader("Upload Cow Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    st.success(f"Predicted Breed: {class_names[predicted.item()]}")
    st.info(f"Confidence: {confidence.item()*100:.2f}%")