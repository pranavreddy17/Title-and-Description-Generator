import gradio as gr
from transformers import pipeline

# Load the model once
generator = pipeline("text2text-generation", model="google/flan-t5-small")

def generate_metadata(script):
    # Generate Title
    title_prompt = f"Generate a catchy YouTube title for this video script:\n{script}"
    title_result = generator(title_prompt, max_new_tokens=20)[0]["generated_text"].strip()

    # Generate Description
    desc_prompt = f"Write a short, engaging YouTube description for this video script:\n{script}"
    desc_result = generator(desc_prompt, max_new_tokens=100)[0]["generated_text"].strip()

    return title_result, desc_result

iface = gr.Interface(
    fn=generate_metadata,
    inputs=gr.Textbox(lines=10, label="🎥 Paste YouTube Script or Summary"),
    outputs=[
        gr.Textbox(label="🏷️ Title"),
        gr.Textbox(label="📄 Description")
    ],
    title="🎬 YouTube Title and Description Generator",
    description="Paste your video content and get a catchy YouTube title and description using Hugging Face transformers."
)

if __name__ == "__main__":
    iface.launch()
