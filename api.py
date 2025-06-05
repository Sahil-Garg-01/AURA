from fastapi import FastAPI
from confluent_kafka import Consumer
import onnxruntime as ort
import numpy as np
from feature_extraction import extract_audio_features, extract_face_features, extract_text_features
import os

app = FastAPI()
session = ort.InferenceSession('fusion_model.onnx')

consumer_conf = {
    'bootstrap.servers': f"{os.getenv('HOST')}:{os.getenv('PORT')}",  
    'security.protocol': 'SSL',
    'ssl.ca.location': os.path.join(os.getcwd(), 'ca.pem'),
    'ssl.certificate.location': os.path.join(os.getcwd(), 'service.cert'),
    'ssl.key.location': os.path.join(os.getcwd(), 'service.key'),
    'group.id': 'emotion-group',
    'auto.offset.reset': 'earliest'
}
consumer = Consumer(consumer_conf)
consumer.subscribe(['emotion-inputs'])

@app.get("/predict")
async def predict_emotion(text: str):
    try:
        msg = consumer.poll(1.0)
        if msg and not msg.error():
            audio_emb = extract_audio_features("sample.wav")
            face_emb = extract_face_features("sample.jpg")
        else:
            audio_emb = np.zeros((1, 2048), dtype=np.float32)
            face_emb = np.zeros((1, 128), dtype=np.float32)
        text_emb = extract_text_features(text if text else "")
        inputs = {'audio_input': audio_emb, 'face_input': face_emb, 'text_input': text_emb}
        prediction = session.run(None, inputs)[0]
        emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        return {'emotion': emotions[np.argmax(prediction)]}
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)