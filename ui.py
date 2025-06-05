import gradio as gr
import requests
import gradio.components as gc

# Verify gradio version
assert gr.__version__ == '4.31.0', f"Expected gradio 4.31.0, got {gr.__version__}"

def predict_emotion(audio, image, text):
    try:
        response = requests.get(f"http://localhost:8000/predict?text={text}")
        return response.json().get('emotion', response.json().get('error', 'Error occurred'))
    except Exception as e:
        return f"UI Error: {str(e)}"

iface = gr.Interface(
    fn=predict_emotion,
    inputs=[
        gc.Audio(type="filepath", label="Audio Input"),
        gc.Image(type="filepath", label="Image Input"),
        gc.Textbox(label="Text Input")
    ],
    outputs="text",
    title="Real-Time Multimodal Emotion Intelligence System"
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)