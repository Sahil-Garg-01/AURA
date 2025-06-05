import numpy as np
import librosa
import cv2
from panns_inference import AudioTagging
from deepface import DeepFace
from transformers import DistilBertTokenizer, TFDistilBertModel
import transformers


try:
    audio_model = AudioTagging(checkpoint_path=None, device='cpu')  # Use GPU if available
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    text_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
except Exception as e:
    print(f"Model initialization failed: {e}")
    raise


def extract_audio_features(audio_path):
    try:
        audio, sr = librosa.load(audio_path, sr=32000, mono=True)
        audio = audio[None, :]  # Add batch dimension
        _, embedding = audio_model.inference(audio)  # Shape: (1, 2048)
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"Audio feature extraction failed: {e}")
        return np.zeros((1, 2048), dtype=np.float32)


def extract_face_features(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or invalid")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # DeepFace expects RGB
        embedding = DeepFace.represent(image, model_name='Facenet', detector_backend='opencv')[0]["embedding"]
        return np.array(embedding).astype(np.float32)  # Shape: (128,)
    except Exception as e:
        print(f"Face feature extraction failed: {e}")
        return np.zeros(128, dtype=np.float32)

def extract_text_features(text):
    try:
        inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=128)
        outputs = text_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return outputs.last_hidden_state[:, 0, :].numpy().astype(np.float32)  # CLS token, Shape: (1, 768)
    except Exception as e:
        print(f"Text feature extraction failed: {e}")
        return np.zeros((1, 768), dtype=np.float32)

if __name__ == "__main__":
    try:
        audio_emb = extract_audio_features("sample.wav")
        face_emb = extract_face_features("sample.jpg")
        text_emb = extract_text_features("")
        print("Audio embedding shape:", audio_emb.shape)
        print("Face embedding shape:", face_emb.shape)
        print("Text embedding shape:", text_emb.shape)
    except Exception as e:
        print(f"Test failed: {e}")