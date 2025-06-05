**Real-Time Multimodal Emotion Intelligence System**


**Overview**
This project is a real-time emotion detection system that analyzes text, audio, and video inputs to predict emotions (e.g., happy, sad, angry). It integrates advanced AI models and streaming technologies to provide robust sentiment analysis, suitable for applications like customer support, telehealth, and marketing.
Key Features

Multimodal Analysis: Processes text (DistilBERT), audio (PANN), and facial expressions (DeepFace).
Real-Time Streaming: Simulates live audio/video streaming via Kafka (WebRTC-ready for production).
Flexible Inputs: Supports partial inputs (e.g., text only) with fallback logic.
Efficient Inference: Uses ONNX for cross-platform model deployment.
User-Friendly Interface: Interactive Gradio UI served via FastAPI.

**Technologies**

1. Machine Learning: TensorFlow, Transformers (DistilBERT), DeepFace, PANN, PyTorch
2. Streaming: Kafka (simulates WebRTC)
3. API & UI: FastAPI, Gradio
4. Model Export: ONNX, tf2onnx
5. Dependencies: Python 3.10.12, NumPy, OpenCV, Librosa, PyTorch

Setup Instructions
Prerequisites

Python 3.10.12: Download from python.org. Ensure it’s added to PATH.
Kafka: Make account (optional for streaming tests). See Apache Kafka.
Windows: Visual Studio Build Tools (for C++ dependencies, optional if using DeepFace).
VS Code: Recommended for development with PowerShell.

**Installation**

Clone the Repository 

**Create a Virtual Environment:**
python -m venv env
.\env\Scripts\activate


**requirements.txt**
tensorflow==2.14.0
panns-inference==0.1.1
deepface==0.0.79
transformers==4.36.0
onnxruntime==1.16.3
fastapi==0.103.0
gradio==4.31.0
confluent-kafka==2.3.0
numpy==1.24.3
librosa==0.10.1
opencv-python==4.8.0.76
tf2onnx==1.16.1
huggingface-hub==0.19.3
tokenizers==0.15.0
uvicorn==0.23.2
torch==2.0.1


**Install:**
pip install -r requirements.txt


**Verify Installation:**
python -c "import tensorflow, transformers, gradio, deepface, panns_inference, cv2; print(tensorflow.__version__, transformers.__version__, gradio.__version__, cv2.__version__)"

Expected: 2.14.0 4.36.0 4.31.0 4.8.0


**Prepare Sample Files:**

Place sample.wav (audio) and sample.jpg (image) in the project directory for testing.


Start Kafka (optional)

**Project Structure**

1. feature_extraction.py: Extracts embeddings from text (DistilBERT), audio (PANN), and images (DeepFace).
2. fusion_model.py: Defines a TensorFlow neural network to fuse embeddings for emotion classification.
3. export_to_onnx.py: Converts the TensorFlow model to ONNX for efficient inference.
4. stream_inputs.py: Simulates WebRTC streaming by sending audio/image data to Kafka.
5. api.py: FastAPI endpoint for emotion predictions using ONNX model and Kafka inputs.
6. ui.py: Gradio interface for uploading inputs and viewing predictions.
7. requirements.txt: Dependency list.

Usage

Run the API:
uvicorn api:app --host 0.0.0.0 --port 8000


Run the Gradio UI:
python ui.py


Open http://localhost:7860 in a browser.
Upload sample.wav, sample.jpg, and enter text (e.g., “I am happy”).
View predicted emotion (e.g., happy, sad).


Test Streaming (optional):
python stream_inputs.py


Sends sample audio/image to Kafka (requires Kafka running).


**Partial Inputs:**

The system supports predictions with text only or text + audio/image. Missing inputs use zeroed embeddings (less accurate).



**Workflow**

1. Input Capture: WebRTC (simulated by stream_inputs.py) streams audio/video to Kafka.
2. Feature Extraction: feature_extraction.py generates embeddings (text: 768D, audio: 2048D, image: 128D).
3. Fusion: fusion_model.py combines embeddings for emotion prediction (7 classes).
4. ONNX Inference: export_to_onnx.py enables efficient prediction via api.py.
5. Serving: api.py provides predictions, visualized by ui.py.