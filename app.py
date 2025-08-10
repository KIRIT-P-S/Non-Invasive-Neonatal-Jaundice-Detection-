import streamlit as st
import numpy as np
import cv2
from PIL import Image
import joblib

st.set_page_config(page_title="Neonatal Jaundice Detection", layout="wide")


rf_model = joblib.load("jaundice_rf_model.pkl")  # Ensure same model as trained


def white_balance_grayworld(img):
    img = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3
    img[:, :, 0] = np.clip(img[:, :, 0] * (avg_gray / avg_b), 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * (avg_gray / avg_g), 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * (avg_gray / avg_r), 0, 255)
    return img.astype(np.uint8)

def retinex_enhancement(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def detect_skin_hybrid(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_mask = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    combined_mask = cv2.bitwise_and(y_mask, h_mask)
    return cv2.bitwise_and(img, img, mask=combined_mask), combined_mask

def clean_skin_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

def brightness_filter(image, mask, low=30, high=220):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.inRange(gray, low, high)
    return cv2.bitwise_and(mask, mask, mask=bright_mask)

def largest_contour_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    result = np.zeros_like(mask)
    cv2.drawContours(result, [largest], -1, 255, thickness=cv2.FILLED)
    return result

def is_valid_detection(mask, min_area=500):
    return cv2.countNonZero(mask) > min_area

def preprocess_image_cv(img):
    img = cv2.resize(img, (224, 224))
    img = white_balance_grayworld(img)
    img = retinex_enhancement(img)
    _, skin_mask = detect_skin_hybrid(img)
    skin_mask = clean_skin_mask(skin_mask)
    skin_mask = brightness_filter(img, skin_mask)
    skin_mask = largest_contour_mask(skin_mask)
    if not is_valid_detection(skin_mask):
        return img
    img = cv2.bitwise_and(img, img, mask=skin_mask)
    return img


from scipy.stats import skew, kurtosis

def extract_features(img_rgb):
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    features = []

    for i in range(3):  
        channel = img_rgb[:, :, i].ravel()
        features.extend([
            np.mean(channel),
            np.std(channel),
            skew(channel),
            kurtosis(channel)
        ])

    for i in range(3): 
        channel = img_hsv[:, :, i].ravel()
        features.extend([
            np.mean(channel),
            np.std(channel),
            skew(channel),
            kurtosis(channel)
        ])

    return np.array(features).reshape(1, -1)

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image("banner.jpg", use_container_width=True)

with col2:
    st.markdown("""
    <h2 style='text-align: center; color: #2E86C1;'>
    Non-Invasive Neonatal Jaundice Detection by Machine Learning with Advanced Image Preprocessing
    </h2>
    <h4 style='text-align: center;'>Kirit P S, Sumathi Shanmuganandam, Praveena A, and Brindha Senthil Kumar</h4>
    <h5 style='text-align: center;'>
    Center of Excellence in Artificial Intelligence and Machine Learning,<br>
    Department of CSE(AI&ML),<br>
    Sri Eswar College of Engineering,<br>
    Kondampatti, Kinathukadavu, Coimbatore, Tamil Nadu 641202, India
    </h5>
    """, unsafe_allow_html=True)

with col3:
    st.image("banner2.jpg", use_container_width=True)

st.markdown("""
<p style='font-style: italic; 
           background: -webkit-linear-gradient(left, #16A085, #00BFFF); 
           -webkit-background-clip: text; 
           -webkit-text-fill-color: transparent;
           font-size: 18px; 
           text-align: center;'>
Revolutionizing newborn care with AI-powered, non-invasive neonatal jaundice detection using advanced image preprocessing for fast, accurate, and contactless diagnosis.
</p>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a baby face image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_cv = np.array(image)[:, :, ::-1]
    img_cv = preprocess_image_cv(img_cv)

    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    features = extract_features(img_rgb)

    pred = rf_model.predict(features)[0]
    confidence = rf_model.predict_proba(features)[0][pred]

    label_map = {0: "Normal", 1: "Jaundice Suspected"}
    st.subheader(f"Prediction: **{label_map[pred]}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
