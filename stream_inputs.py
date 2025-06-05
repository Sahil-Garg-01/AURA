from confluent_kafka import Producer
import cv2
import numpy as np
import librosa
import json
import os

# Kafka producer setup
conf = {
    'bootstrap.servers': f"{os.getenv('HOST')}:{os.getenv('PORT')}",
    'security.protocol': 'SSL',
    'ssl.ca.location': os.path.join(os.getcwd(), 'ca.pem'),
    'ssl.certificate.location': os.path.join(os.getcwd(), 'service.cert'),
    'ssl.key.location': os.path.join(os.getcwd(), 'service.key')
}
producer = Producer(conf)

def deliver_callback(err, msg):
    if err:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()}')

def stream_inputs(audio_path, image_path):
    try:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or invalid")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = {
            'audio': audio.tolist()[:16000],
            'image': image.flatten().tolist()[:10000]
        }
        producer.produce('emotion-inputs', json.dumps(data), callback=deliver_callback)
        producer.flush()
    except Exception as e:
        print(f"Streaming failed: {e}")

if __name__ == "__main__":
    stream_inputs("sample.wav", "sample.jpg")