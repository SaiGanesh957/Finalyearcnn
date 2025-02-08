import os
import gdown
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU only
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)

# Paths
MODEL_PATH = 'models/vgg16_model.h5'
UPLOAD_DIR = 'static/images'
DATASET_DIR = 'static/dataset'
FEATURES_PATH = 'models/dataset_features.npy'
IMAGES_PATH = 'models/dataset_images.npy'


if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    model_url = "https://drive.google.com/file/d/1-Uu7V0t7qhfNLty4e48j5GewXcUd-voO/view?usp=sharing"
    gdown.download(model_url, MODEL_PATH, fuzzy=True, quiet=False)

# Load VGG16 Model
model = tf.keras.models.load_model(MODEL_PATH)

# Feature Extraction Function
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)[np.newaxis, ...]
    
    feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    features = feature_extractor.predict(img_array, verbose=0).flatten()  # Suppress output
    return features

# Load or Compute Dataset Features
if os.path.exists(FEATURES_PATH) and os.path.exists(IMAGES_PATH):
    print("🔵 Loading precomputed features...")
    dataset_features = np.load(FEATURES_PATH)
    dataset_images = np.load(IMAGES_PATH, allow_pickle=True).tolist()
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

    # Save extracted features to avoid recomputing next time
    np.save(FEATURES_PATH, dataset_features)
    np.save(IMAGES_PATH, np.array(dataset_images, dtype=object))
    print("✅ Features saved successfully!")

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
    
    # Extract features of uploaded image
    query_features = extract_features(filepath).reshape(1, -1)
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_features, dataset_features)
    top_indices = similarities.argsort()[0][-5:][::-1]  # Top 5 most similar images
    
    recommended_images = [dataset_images[i] for i in top_indices]
    
    return jsonify({
        'uploaded_image': file.filename,
        'recommended_images': recommended_images
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT dynamically
    app.run(host="0.0.0.0", port=port, debug=False)

