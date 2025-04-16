import gradio as gr
import os
# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .llm import LLMHandler
from .router import ModelRouter

if not os.getenv('HF_TOKEN'):
    print("Warning: HF_TOKEN environment variable not set. Please set it before running the app.")
    print("Example: export HF_TOKEN='your_huggingface_token_here'")

router = ModelRouter()
llm_handler = LLMHandler()

model_descriptions = {
    "Gemma 3 27B": "Great for most questions",
    "Llama 3.2 1B": "Fast at basic tasks",
    "Distill R1 1.5B": "Compact and efficient",
    "Distill R1 32B": "Powerful for reasoning tasks"
}

def respond(message, chat_history, model_name, notification):
    if not message:
        return "", chat_history, notification

    history_tuples = []
    for msg in chat_history:
        if msg["role"] == "user":
            history_tuples.append((msg["content"], ""))
        elif msg["role"] == "assistant" and history_tuples:
            history_tuples[-1] = (history_tuples[-1][0], msg["content"])

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})

    response_generator = llm_handler.generate_streaming_response(
        message, 
        history_tuples, 
        model_name
    )

    first_chunk = next(response_generator)
    if first_chunk.startswith("Using "):
        print(f"[DEBUG] {first_chunk}")
        model_notification = f"🤖 {first_chunk}"
        selected_model = first_chunk.replace("Using ", "").split(" (")[0]
        gr.Info(f"🤖 Model selected: {selected_model}")

        for chunk in response_generator:
            chat_history[-1]["content"] = chunk
            yield "", chat_history, model_notification
    else:
        if first_chunk.startswith("Sorry, the") and ("service is currently unavailable" in first_chunk):
            model_notification = """<div class=\"error-notification\">❌ Service Unavailable - Please try a different model or try again later</div>"""
            print(f"[ERROR] Service unavailable detected")
            gr.Warning("❌ Service Unavailable - The model service is not responding")
            chat_history[-1]["content"] = first_chunk
            yield "", chat_history, model_notification
            return
        elif model_name == "automatic":
            model_notification = "🤖 Using automatic model selection (no model info returned)"
            print("[DEBUG] Automatic selection, but no model info returned")
            gr.Info("🤖 Using automatic model selection")
        else:
            model_notification = f"🤖 Using {model_name} (manually selected)"
            print(f"[DEBUG] Manually selected model: {model_name}")
            gr.Info(f"🤖 Model selected: {model_name}")

        chat_history[-1]["content"] = first_chunk
        yield "", chat_history, model_notification

        for chunk in response_generator:
            chat_history[-1]["content"] = chunk
            yield "", chat_history, model_notification, ""

# UI

custom_css = """
body {
    background-color: #1e1e1e;
    color: white;
    font-family: 'Inter', sans-serif;
    margin: 0;
}
#main-container {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100vh;
    padding: 1rem;
}
#header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 1rem;
}
#header .gr-box {
    flex: 1;
    display: flex;
    justify-content: flex-start;
}
#header-center {
    flex: 1;
    text-align: center;
    font-size: 1.5em;
    font-weight: bold;
}
#chat-area {
    flex: 1;
    background-color: #2a2a2a;
    border-radius: 10px;
    padding: 1rem;
    overflow-y: auto;
    margin-bottom: 1rem;
}
#input-wrapper {
    position: relative;
    display: flex;
    align-items: center;
    width: 100%;
}
#input-box textarea {
    background-color: #2a2a2a;
    color: white;
    border: 1px solid #444;
    border-radius: 20px;
    resize: none;
    font-size: 16px;
    width: 100%;
    padding: 1rem 3rem 1rem 1rem;
    box-sizing: border-box;
}
#send-btn {
    position: absolute;
    right: 12px;
    bottom: 12px;
    background-color: #10a37f;
    border-radius: 50%;
    width: 36px;
    height: 36px;
    display: flex;
    justify-content: center;
    align-items: center;
    border: none;
    color: white;
    font-size: 16px;
    cursor: pointer;
}
.dropdown-style select option[data-description]:after {
    content: attr(data-description);
    display: block;
    font-size: 0.8em;
    color: #aaa;
}
.notification-area {
    font-size: 0.9em;
    color: #ccc;
    margin-top: 8px;
    text-align: center;
}
"""

with gr.Blocks(css="#main-container") as demo:
    gr.HTML(f"<style>{custom_css}</style>")

    with gr.Column():
        with gr.Row(elem_id="header"):
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=["automatic"] + [
                        f"{name} ({desc})" for name, desc in model_descriptions.items()
                    ],
                    value="automatic",
                    label=None,
                    container=False,
                    elem_classes=["dropdown-style"]
                )
            with gr.Column(scale=2):
                gr.Markdown("""<div id='header-center'>LLM Router</div>""")

        chatbot = gr.Chatbot(
            label=None,
            height=500,
            elem_id="chat-area",
            show_copy_button=True,
            type="messages"
        )

        with gr.Row():
            with gr.Column(elem_id="input-wrapper"):
                msg = gr.Textbox(
                    placeholder="Ask anything",
                    show_label=False,
                    lines=1,
                    elem_id="input-box"
                )
                send_btn = gr.Button("\u27A4", elem_id="send-btn")

        model_notification = gr.Markdown("", elem_id="notification", elem_classes=["notification-area"])

        send_btn.click(
            fn=respond,
            inputs=[msg, chatbot, model_dropdown, model_notification],
            outputs=[msg, chatbot, model_notification],
            queue=True
        )

        msg.submit(
            fn=respond,
            inputs=[msg, chatbot, model_dropdown, model_notification],
            outputs=[msg, chatbot, model_notification],
            queue=True
        )

        gr.Button("🧹 Clear chat").click(
            fn=lambda: ([], ""),
            outputs=[chatbot, model_notification],
            queue=False
        )
