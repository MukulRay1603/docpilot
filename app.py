# HuggingFace Spaces entry point — imports and launches the Gradio UI.
# Set GROQ_API_KEY as a Space secret in Settings → Repository secrets.
from gradio_app import demo

demo.launch()
