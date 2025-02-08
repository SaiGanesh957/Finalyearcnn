import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import gdown

# 🚀 Disable GPU & Hide Warnings (Important for Railway)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow warnings

# Initialize Flask App
app = Flask(__name__)

# 🔹 Define Paths
MODEL_PATH = "/tmp/vgg16_model.h5"  # Use /tmp for Railway
UPLOAD_DIR = "/tmp"  # Railway storage for uploaded images
DATASET_DIR = "static/dataset"  # Update if dataset location changes

# 🔹 Google Drive Model Download (Replace with your own file ID)
MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"

# 🔹 Download Model from Google Drive if Not Found
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, fuzzy=True, quiet=False)

# ✅ Load Model
print("🔵 Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("✅ Model Loaded Successfully!")

# 🔹 Feature Extraction Function
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)[np.newaxis, ...]
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    features = feature_extractor.predict(img_array, verbose=0).flatten()
    return features

# 🔹 Load Precomputed Features (to avoid reprocessing dataset on each restart)
FEATURES_FILE = "/tmp/dataset_features.npy"
IMAGES_FILE = "/tmp/dataset_images.npy"

if os.path.exists(FEATURES_FILE) and os.path.exists(IMAGES_FILE):
    print("🔵 Loading precomputed features...")
    dataset_features = np.load(FEATURES_FILE)
    dataset_images = np.load(IMAGES_FILE, allow_pickle=True).tolist()
else:
    print("🟡 Extracting dataset features (this will take time only once)...")
    dataset_features = []
    dataset_images = []

    for img_file in os.listdir(DATASET_DIR):
        img_path = os.path.join(DATASET_DIR, img_file)
        features = extract_features(img_path)
        dataset_features.append(features)
        dataset_images.append(img_file)

    dataset_features = np.array(dataset_features)
    np.save(FEATURES_FILE, dataset_features)
    np.save(IMAGES_FILE, np.array(dataset_images, dtype=object))
    print("✅ Features saved successfully!")

# ✅ Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)

    # 🔹 Extract features of uploaded image
    query_features = extract_features(filepath).reshape(1, -1)

    # 🔹 Calculate cosine similarity
    similarities = cosine_similarity(query_features, dataset_features)
    top_indices = similarities.argsort()[0][-5:][::-1]  # Top 5 most similar images

    recommended_images = [dataset_images[i] for i in top_indices]

    return jsonify({
        'uploaded_image': file.filename,
        'recommended_images': recommended_images
    })

# ✅ Start Flask App with Correct Port (For Railway)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Railway assigns a dynamic port
    app.run(host="0.0.0.0", port=port, debug=False)
