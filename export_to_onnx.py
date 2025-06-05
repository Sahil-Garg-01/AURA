import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort
from fusion_model import build_fusion_model

# Build and save TensorFlow model
model = build_fusion_model()
model.save('fusion_model')

# Convert to ONNX
spec = [
    tf.TensorSpec((None, 2048), tf.float32, name='audio_input'),
    tf.TensorSpec((None, 128), tf.float32, name='face_input'),
    tf.TensorSpec((None, 768), tf.float32, name='text_input')
]
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path='fusion_model.onnx')

# Test ONNX model
session = ort.InferenceSession('fusion_model.onnx')
audio_emb = np.random.rand(1, 2048).astype(np.float32)
face_emb = np.random.rand(1, 128).astype(np.float32)
text_emb = np.random.rand(1, 768).astype(np.float32)
inputs = {'audio_input': audio_emb, 'face_input': face_emb, 'text_input': text_emb}
outputs = session.run(None, inputs)
print("ONNX prediction shape:", outputs[0].shape)