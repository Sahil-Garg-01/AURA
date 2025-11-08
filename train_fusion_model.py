import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from fusion_model import build_fusion_model
import json

# 1️ Define emotion labels and mapping
emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
label_to_index = {label: i for i, label in enumerate(emotions)}

# 2️ Load or prepare training data (example placeholders)
# Each of these should be numpy arrays of embeddings
# For example, shape: (num_samples, embedding_dim)
audio_features = np.load('audio_embeddings.npy')   # shape: (N, 2048)
face_features = np.load('face_embeddings.npy')     # shape: (N, 128)
text_features = np.load('text_embeddings.npy')     # shape: (N, 768)
labels = np.load('labels.npy')                     # shape: (N,)

# Convert labels to one-hot encoding
num_classes = len(emotions)
labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# 3️ Split into train/test
X_audio_train, X_audio_val, X_face_train, X_face_val, X_text_train, X_text_val, y_train, y_val = train_test_split(
    audio_features, face_features, text_features, labels_onehot, test_size=0.2, random_state=42
)

# 4️ Build the fusion model
model = build_fusion_model(
    audio_dim=2048,
    face_dim=128,
    text_dim=768,
    num_classes=num_classes
)

# 5️ Train the model
history = model.fit(
    [X_audio_train, X_face_train, X_text_train],
    y_train,
    validation_data=([X_audio_val, X_face_val, X_text_val], y_val),
    epochs=20,
    batch_size=32
)

# 6️ Save trained model
model.save('trained_fusion_model')

# 7️ Save emotion label mapping for inference
with open('emotion_labels.json', 'w') as f:
    json.dump(emotions, f)

print("Training complete. Model and label mapping saved.")
