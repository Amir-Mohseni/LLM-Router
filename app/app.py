import gradio as gr
import os
# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .llm import LLMHandler
from .router import ModelRouter

# Check for API key
if not os.getenv('HF_TOKEN'):
    print("Warning: HF_TOKEN environment variable not set. Please set it before running the app.")
    print("Example: export HF_TOKEN='your_huggingface_token_here'")

# Initialize the router and LLM handler
router = ModelRouter()
llm_handler = LLMHandler()

def respond(message, chat_history, model_name, notification):
    """
    Process the user message and get a streaming response from the selected LLM
    """
    if not message:
        return "", chat_history, notification
    
    # Create tuple-based history for the LLM handler (from messages format)
    history_tuples = []
    for msg in chat_history:
        if msg["role"] == "user":
            # Start a new conversation turn
            history_tuples.append((msg["content"], ""))
        elif msg["role"] == "assistant" and history_tuples:
            # Complete the last turn with the assistant's response
            history_tuples[-1] = (history_tuples[-1][0], msg["content"])
    
    # Add user message to history
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})
    
    # Return intermediate states to update the UI
    response_generator = llm_handler.generate_streaming_response(
        message, 
        history_tuples, 
        model_name
    )
    
    # First value might be model info
    first_chunk = next(response_generator)
    if first_chunk.startswith("Using "):
        # Both print to terminal and update notification area
        print(f"[DEBUG] {first_chunk}")
        model_notification = f"🤖 {first_chunk}"
        
        # Show pop-up notification
        selected_model = first_chunk.replace("Using ", "").split(" (")[0]
        gr.Info(f"🤖 Model selected: {selected_model}")
        
        # Continue with actual response chunks
        for chunk in response_generator:
            chat_history[-1]["content"] = chunk
            yield "", chat_history, model_notification
    else:
        # Handle error messages (including service unavailable)
        if first_chunk.startswith("Sorry, the") and ("service is currently unavailable" in first_chunk or "Hugging Face service is currently unavailable" in first_chunk):
            model_notification = """<div class="error-notification">❌ Service Unavailable - Please try a different model or try again later</div>"""
            print(f"[ERROR] Service unavailable detected")
            # Show error pop-up
            gr.Warning("❌ Service Unavailable - The model service is not responding")
            
            # Add error message with special marker for CSS
            chat_history[-1]["content"] = first_chunk
            yield "", chat_history, model_notification
            return
        # If no model info (manual selection or other errors), create notification for the selected model
        elif model_name == "automatic":
            model_notification = "🤖 Using automatic model selection (no model info returned)"
            print("[DEBUG] Automatic selection, but no model info returned")
            # Show generic pop-up for automatic selection
            gr.Info("🤖 Using automatic model selection")
        else:
            model_notification = f"🤖 Using {model_name} (manually selected)"
            print(f"[DEBUG] Manually selected model: {model_name}")
            # Show pop-up for manual selection
            gr.Info(f"🤖 Model selected: {model_name}")
            
        # First chunk is actual response content
        chat_history[-1]["content"] = first_chunk
        yield "", chat_history, model_notification
        
        # Continue with rest of chunks
        for chunk in response_generator:
            chat_history[-1]["content"] = chunk
            yield "", chat_history, model_notification, ""

# Create Gradio interface
with gr.Blocks(title="Streaming LLM Chat App") as demo:
    gr.Markdown("# 📡Router")
    gr.Markdown("Select a model or use automatic mode to chat with different language models. Watch responses stream in real-time!")
    
    chatbot = gr.Chatbot(
        label="Conversation",
        # label="",
        height=650,
        show_copy_button=True,
        elem_id="chatbot",
        type="messages"
    )
    
    with gr.Row():
        with gr.Column(scale=6):
            msg = gr.Textbox(
                label="Your message",
                placeholder="Type your message here...",
                lines=2,
                show_label=False,
                container=False
            )
        with gr.Column(scale=1, min_width=100):
            model_dropdown = gr.Dropdown(
                choices=["automatic"] + router.get_available_models(),
                value="automatic",
                label="Select LLM",
                elem_id="model_dropdown",
                show_label=False,
                container=True
            )
        with gr.Column(scale=1, min_width=100):
            submit_btn = gr.Button("Send", variant="primary",elem_id="send",)
        with gr.Column(scale=1, min_width=100):
            clear_btn = gr.Button("Clear chat",elem_id="clear",)
    
    # Add a notification component at the bottom
    with gr.Row():
        model_notification = gr.Markdown(
            "", 
            elem_id="model-notification",
            elem_classes=["notification-area"]
        )
    
    # Examples
    gr.Examples(
        examples=[
            ["Tell me a short story about a robot discovering emotions."],
            ["What are the key differences between quantum computing and classical computing?"],
            ["Give me a simple recipe for chocolate chip cookies."]
        ],
        inputs=msg,
        elem_id="centered-examples"
    )
    
    # CSS for the notification
    css = """
    #component-17{
        display: none !important;;
    }
    footer{
        margin: 0 !important;
    }
    #send, #clear  {
        margin: 10px 0px;
    } 
    #centered-examples {
        display: flex;
        flex-direction: row;
        align-items: center;         /* Vertically align children */
        justify-content: center;     /* Horizontally center everything */
        gap: 16px;                   /* Space between heading and buttons */
    }   

    #centered-examples > div:first-child {
        display: flex;
        align-items: center;
        text-align: center;
        white-space: nowrap;         /* Prevent line breaks */
        height: 100%;                /* Fill the height for vertical centering */
    }

    #centered-examples > div:last-child {
        display: flex !important;
        align-items: center;         /* Vertically center example buttons */
        justify-content: center !important;
        flex-wrap: wrap;
        gap: 8px;
    }

    .notification-area {
        padding: 0px 15px 0px 15px;
        border-radius: 8px;
        background-color: rgba(0, 0, 0, 0.05);
        transition: opacity 0.5s ease;
        font-size: 0.9em;
    }
    .notification-area:empty {
        opacity: 0;
        padding: 0;
        margin: 0;
    }
    /* Style for pop-up notifications */
    .gradio-info {
        font-weight: 500 !important;
        font-size: 1.05em !important;
    }
    /* Style for error notifications */
    .error-notification {
        background-color: rgba(220, 38, 38, 0.1);
        color: rgba(185, 28, 28, 1);
        font-weight: 500;
        border-left: 4px solid rgba(220, 38, 38, 0.8);
        padding: 8px 15px;
        border-radius: 8px;
    }
    """
    
    # Add the CSS to the page
    gr.HTML(f"<style>{css}</style>")
    
    # Set up event handlers
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, model_dropdown, model_notification],
        outputs=[msg, chatbot, model_notification],
        queue=True
    )
    
    submit_btn.click(
        fn=respond,
        inputs=[msg, chatbot, model_dropdown, model_notification],
        outputs=[msg, chatbot, model_notification],
        queue=True
    )
    
    # Update clear function to also clear the notification
    clear_btn.click(
        fn=lambda: ([], ""),
        outputs=[chatbot, model_notification],
        queue=False
    )

# No longer need the launch code here as it's in main.py 