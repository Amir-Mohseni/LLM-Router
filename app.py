import gradio as gr
import os
from llm import LLMHandler
from router import ModelRouter

# Check for API key
if not os.getenv('HF_TOKEN'):
    print("Warning: HF_TOKEN environment variable not set. Please set it before running the app.")
    print("Example: export HF_TOKEN='your_huggingface_token_here'")

# Initialize the router and LLM handler
router = ModelRouter()
llm_handler = LLMHandler()

# Chat history to maintain the conversation
chat_history = []

def respond(message, history, model_name):
    """Process the user message and get a response from the selected LLM"""
    if not message:
        return history
    
    # Get response from the LLM handler
    updated_history = llm_handler.generate_response(message, history.copy() if history else [], model_name)
    
    return updated_history

# Create Gradio interface
with gr.Blocks(title="LLM Chat App") as demo:
    gr.Markdown("# 💬 Chat with Hugging Face LLMs")
    gr.Markdown("Select a model or use automatic mode to chat with different language models")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=400,
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Type your message",
                    placeholder="Ask me anything...",
                    lines=2,
                    show_label=False,
                    container=False
                )
                
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["automatic"] + router.get_available_models(),
                    value="automatic",
                    label="Select LLM"
                )
                submit_btn = gr.Button("Send", variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("Clear conversation")
    
    # Set up interactions
    submit_btn.click(
        fn=respond,
        inputs=[msg, chatbot, model_dropdown],
        outputs=[chatbot],
        queue=True
    ).then(
        fn=lambda: "", 
        outputs=[msg]
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, model_dropdown],
        outputs=[chatbot],
        queue=True
    ).then(
        fn=lambda: "", 
        outputs=[msg]
    )
    
    clear_btn.click(lambda: [], None, chatbot, queue=False)
    
    # Add some example inputs
    gr.Examples(
        examples=[
            ["Tell me a short story about a cat"],
            ["What is the capital of France?"],
            ["Explain quantum computing in simple terms"]
        ],
        inputs=msg
    )

# Launch the app
if __name__ == "__main__":
    demo.launch() 