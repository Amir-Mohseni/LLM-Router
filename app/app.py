import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv(".env.example")

print("OPENAI_API_KEY =", os.getenv("OPENAI_API_KEY"))

from .llm import LLMHandler
from .router import ModelRouter
from RouteLLM.route_llm_classifier import RouteLLMClassifier

# Initialize the router and LLM handler
router = ModelRouter()
llm_handler = LLMHandler()
routeLLM = RouteLLMClassifier("gpt-4","gpt-3.5-turbo")

def respond(message, chat_history, model_name, notification, reasoning_state, has_reasoning_state):
    """
    Process the user message and get a streaming response from the selected LLM
    """
    if not message:
        return "", chat_history, notification, reasoning_state, has_reasoning_state
    
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
    
    # Initialize state for streaming
    thinking_buffer = ""
    response_buffer = ""
    thinking_complete = False
    selected_model = ""
    
    # Return intermediate states to update the UI
    if model_name != "RouteLLM":
        response_generator = llm_handler.generate_streaming_response(
            message, 
            history_tuples, 
            model_name,
            enable_reasoning=True  # Always try to enable reasoning
        )
    
    # Add initial placeholders for thinking and response
    thinking_msg = {
        "role": "assistant",
        "content": "🧠 Thinking...",
        "metadata": {"title": "⏳ Thinking Process"}
    }
    
    response_msg = {
        "role": "assistant",
        "content": "⌛ Generating response..."
    }
    if model_name == "RouteLLM":
        model_notification = "🤖 Using RouteLLM router"
        response_msg = {
            "role": "assistant",
            "content": "⌛ Generating response..."
        }
        chat_history.append(response_msg)
        response_buffer = ""
        
        # Construct messages with full conversation history
        messages = []
        for user_msg, assistant_msg in history_tuples:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})
        
        try:
            response = routeLLM.client.chat.completions.create(
                model=f"router-mf-{routeLLM.threshold:.5f}",
                messages=messages,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and "content" in chunk.choices[0].delta:
                    response_buffer += chunk.choices[0].delta.content
                    chat_history[-1]["content"] = response_buffer
                    yield "", chat_history, model_notification, "", False
        except Exception as e:
            model_notification = f"⚠️ Error with RouteLLM: {str(e)}"
            yield "", chat_history, model_notification, "", False
    else:
        # Process the first chunk to extract model info or content
        try:
            first_chunk = next(response_generator)
            
            # The response is now a dictionary with a "content" field
            if isinstance(first_chunk, dict):
                content = first_chunk.get("content", "")
                
                # Check if this is model info (automatic selection)
                if content.startswith("Using "):
                    # Both print to terminal and update notification area
                    print(f"[DEBUG] {content}")
                    model_notification = f"🤖 {content}"
                    
                    # Extract the model name for notification
                    selected_model = content.replace("Using ", "").split(" (")[0]
                    gr.Info(f"🤖 Model selected: {selected_model}")
                    
                    # Add response message placeholder for the actual response
                    chat_history.append(response_msg)
                    
                    # Check if reasoning is available from the first chunk
                    has_reasoning = first_chunk.get("has_reasoning", False)
                    reasoning_text = first_chunk.get("reasoning", "")
                    
                    # Create a separate thinking message if reasoning is supported
                    if has_reasoning:
                        # Add a visual indicator that reasoning is available
                        model_notification += " 🧠 Model supports reasoning"
                        thinking_buffer = reasoning_text if reasoning_text else ""
                        
                        # Insert the thinking message BEFORE the response message
                        # This ensures thinking appears as a separate message
                        chat_history.insert(-1, thinking_msg.copy())
                        chat_history[-2]["content"] = thinking_buffer if thinking_buffer else "🧠 Thinking..."
                    
                    # Process remaining chunks
                    for chunk in response_generator:
                        # Extract content and reasoning
                        chunk_content = chunk.get("content", "")
                        chunk_reasoning = chunk.get("reasoning", "")
                        
                        # If we have reasoning content and reasoning is enabled
                        if chunk_reasoning and has_reasoning:
                            # Update thinking buffer
                            thinking_buffer += chunk_reasoning
                            chat_history[-2]["content"] = thinking_buffer
                        
                        # Always update the response buffer with content
                        if chunk_content:
                            response_buffer = chunk_content
                            chat_history[-1]["content"] = response_buffer
                        
                        # Yield the updated state
                        yield "", chat_history, model_notification, thinking_buffer, has_reasoning
                else:
                    # Direct content response (not model info)
                    model_notification = f"🤖 Using {model_name}" if model_name != "automatic" else "🤖 Using automatic model selection"
                    
                    # Check for reasoning in first chunk
                    has_reasoning = first_chunk.get("has_reasoning", False)
                    reasoning_text = first_chunk.get("reasoning", "")
                    
                    # Add response message for the content first
                    response_buffer = content
                    chat_history.append(response_msg)
                    chat_history[-1]["content"] = response_buffer
                    
                    # If reasoning is supported, add a separate thinking message BEFORE the response
                    if has_reasoning or reasoning_text:
                        model_notification += " 🧠 Model supports reasoning"
                        thinking_buffer = reasoning_text if reasoning_text else ""
                        
                        # Insert the thinking message before the response message
                        chat_history.insert(-1, thinking_msg.copy())
                        chat_history[-2]["content"] = thinking_buffer
                    
                    # Yield initial state
                    yield "", chat_history, model_notification, thinking_buffer if has_reasoning else "", has_reasoning
                    
                    # Process remaining chunks
                    for chunk in response_generator:
                        if isinstance(chunk, dict):
                            chunk_content = chunk.get("content", "")
                            chunk_reasoning = chunk.get("reasoning", "")
                            
                            # Handle reasoning updates if available
                            if chunk_reasoning and has_reasoning:
                                # Make sure we have a thinking message
                                if len(chat_history) < 2 or "Thinking Process" not in str(chat_history[-2].get("metadata", {})):
                                    # Insert thinking message if it doesn't exist yet
                                    chat_history.insert(-1, thinking_msg.copy())
                                
                                # Update thinking buffer and message content
                                thinking_buffer += chunk_reasoning
                                chat_history[-2]["content"] = thinking_buffer
                            
                            # Update response content
                            if chunk_content:
                                response_buffer = chunk_content
                                chat_history[-1]["content"] = response_buffer
                        else:
                            # Use content directly
                            if isinstance(chunk, str):
                                response_buffer = chunk
                                chat_history[-1]["content"] = response_buffer
                        
                        # Yield updated state
                        yield "", chat_history, model_notification, thinking_buffer if has_reasoning else "", has_reasoning
            else:
                # Legacy string-based response handling
                model_notification = f"🤖 Using {model_name} or automatic selection"
                response_buffer = first_chunk if isinstance(first_chunk, str) else ""
                thinking_buffer = ""
                has_reasoning = False
                
                # Add response message
                chat_history.append(response_msg)
                chat_history[-1]["content"] = response_buffer
                
                # Yield initial state
                yield "", chat_history, model_notification, "", False
                
                # Process remaining chunks
                for chunk in response_generator:
                    updated = False
                    chunk_content = ""
                    
                    if isinstance(chunk, dict):
                        chunk_content = chunk.get("content", "")
                        chunk_reasoning = chunk.get("reasoning", "")
                        
                        # Handle reasoning if available
                        if chunk_reasoning:
                            # Add thinking message if we don't have one
                            if "Thinking Process" not in str([m.get("metadata", {}) for m in chat_history]):
                                chat_history.insert(-1, thinking_msg.copy())
                                thinking_buffer = chunk_reasoning
                                chat_history[-2]["content"] = thinking_buffer
                                updated = True
                                has_reasoning = True
                        
                        if chunk_content:
                            response_buffer = chunk_content
                            chat_history[-1]["content"] = response_buffer
                            updated = True
                    elif isinstance(chunk, str):
                        response_buffer = chunk
                        chat_history[-1]["content"] = response_buffer
                        updated = True
                    
                    # Yield updated state only if something changed
                    if updated:
                        yield "", chat_history, model_notification, thinking_buffer if has_reasoning else "", has_reasoning
    
        except StopIteration:
            # Handle empty response
            model_notification = "⚠️ No response received from the model"
            yield "", chat_history, model_notification, "", False
        
        except Exception as e:
            # Handle other errors
            model_notification = f"⚠️ Error: {str(e)}"
            print(f"[ERROR] {str(e)}")
            yield "", chat_history, model_notification, "", False

# Create Gradio interface
with gr.Blocks(title="Streaming LLM Chat App") as demo:
    gr.Markdown("# 📡Router")
    gr.Markdown("Select a model or use automatic mode to chat with different language models. Watch responses stream in real-time!")
    
    # State variables for tracking reasoning content
    reasoning_state = gr.State("")
    has_reasoning_state = gr.State(False)
    
    chatbot = gr.Chatbot(
        label="Conversation",
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
                choices=["automatic","RouteLLM"] + router.get_available_models(),
                value="automatic",
                label="Select LLM",
                elem_id="model_dropdown",
                show_label=False,
                container=True
            )
        with gr.Column(scale=1, min_width=100):
            submit_btn = gr.Button("Send", variant="primary", elem_id="send")
        with gr.Column(scale=1, min_width=100):
            clear_btn = gr.Button("Clear chat", elem_id="clear")
    
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
    
    # CSS for the notification and thinking section
    css = """
    /* Styling for thinking blocks */
    .thinking-block {
        background-color: #f0f7ff;
        border-left: 4px solid #2563eb;
        padding: 10px 15px;
        margin: 5px 0;
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 0.9em;
        border-radius: 4px;
    }
    
    .thinking-title {
        font-weight: bold;
        color: #2563eb;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    /* Make thinking blocks collapsible */
    .thinking-content {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.3s ease-out;
    }
    
    .thinking-content.expanded {
        max-height: 800px;
        overflow: auto;
        transition: max-height 0.5s ease-in;
    }
    
    /* Toggle button for thinking blocks */
    .thinking-toggle {
        cursor: pointer;
        user-select: none;
        color: #4b5563;
        font-size: 0.8em;
        padding: 2px 6px;
        border-radius: 4px;
        background: #e5e7eb;
        margin-left: auto;
    }
    
    .thinking-toggle:hover {
        background: #d1d5db;
    }
    
    /* Add special styling for thinking messages in the chat */
    .message[data-testid="thinking-message"],
    .thinking-message,
    .message[data-testid*="thinking"] {
        background-color: #f0f7ff !important;
        border-left: 4px solid #2563eb !important;
        position: relative;
    }
    
    /* Add animation for the thinking indicator */
    @keyframes thinking {
        0% { content: "🧠 ."; }
        33% { content: "🧠 .."; }
        66% { content: "🧠 ..."; }
        100% { content: "🧠 ...."; }
    }
    
    .thinking-indicator:after {
        content: "🧠 ...";
        animation: thinking 1.5s infinite;
        display: inline-block;
    }
    
    /* Other CSS styles... */
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
    
    # Add JavaScript for collapsible thinking sections
    js = """
    function setupThinkingBlocks() {
        // Find all messages
        const messages = document.querySelectorAll('.message');
        
        messages.forEach(msg => {
            // Check if this is a thinking message that hasn't been processed
            const messageText = msg.textContent || '';
            
            // Apply thinking styling to thinking messages (both thinking markers and metadata)
            const isThinking = 
                (messageText.includes("🧠") || messageText.includes("Thinking")) || 
                (msg.getAttribute('data-testid') && msg.getAttribute('data-testid').includes('thinking'));
                
            if (isThinking && !msg.getAttribute('data-thinking-styled')) {
                msg.setAttribute('data-testid', 'thinking-message');
                msg.setAttribute('data-thinking-styled', 'true');
                msg.classList.add('thinking-message');
                console.log('Applied thinking styling to message');
            }
            
            // Specifically look for thinking process blocks to make collapsible
            const hasThinkingProcess = 
                messageText.includes("🧠 Thinking Process") || 
                (msg.querySelector('.message-body') && 
                 msg.querySelector('.message-body').textContent.includes("Thinking Process"));
                
            if (hasThinkingProcess && !msg.classList.contains('thinking-processed')) {
                console.log('Processing thinking block');
                
                // Mark as processed
                msg.classList.add('thinking-processed');
                
                // Create thinking block structure
                const content = msg.querySelector('.message-body');
                if (!content) {
                    console.log('No message body found in thinking message');
                    return;
                }
                
                // Extract the content, removing the header
                const fullText = content.textContent || '';
                const headerEnd = fullText.indexOf("\\n\\n");
                const textContent = headerEnd !== -1 ? 
                    fullText.substring(headerEnd + 2) : 
                    fullText.replace("🧠 Thinking Process", "").trim();
                
                console.log('Extracted thinking content');
                
                // Clear original content
                content.innerHTML = '';
                
                // Create thinking block elements
                const thinkingBlock = document.createElement('div');
                thinkingBlock.className = 'thinking-block';
                
                const title = document.createElement('div');
                title.className = 'thinking-title';
                title.innerHTML = '🧠 <span>Thinking Process</span>';
                
                const toggle = document.createElement('span');
                toggle.className = 'thinking-toggle';
                toggle.textContent = 'Hide';
                toggle.onclick = function() {
                    const content = this.parentNode.nextSibling;
                    const isExpanded = content.classList.toggle('expanded');
                    this.textContent = isExpanded ? 'Hide' : 'Show';
                };
                
                title.appendChild(toggle);
                
                const thinkingContent = document.createElement('div');
                thinkingContent.className = 'thinking-content expanded';
                thinkingContent.textContent = textContent || "The model is thinking...";
                
                // Assemble the thinking block
                thinkingBlock.appendChild(title);
                thinkingBlock.appendChild(thinkingContent);
                content.appendChild(thinkingBlock);
                console.log('Successfully created collapsible thinking block');
            }
        });
        
        // Schedule the next check
        setTimeout(setupThinkingBlocks, 500);  // Check regularly
    }
    
    // Start monitoring for thinking blocks
    document.addEventListener('DOMContentLoaded', function() {
        console.log('DOM loaded, starting thinking block setup');
        setupThinkingBlocks();
    });
    
    // Also run when content might have changed
    const observer = new MutationObserver(function(mutations) {
        setupThinkingBlocks();
    });
    
    // Start observing the chatbot container
    setTimeout(function() {
        const chatbot = document.getElementById('chatbot');
        if (chatbot) {
            observer.observe(chatbot, { childList: true, subtree: true });
        }
    }, 1000);
    """
    
    # Add the CSS and JavaScript to the page
    gr.HTML(f"<style>{css}</style><script>{js}</script>")
    
    # Debugging function to print model capabilities
    def debug_models():
        """Display model capabilities and configurations"""
        print("\nModel capabilities:")
        for model_key, info in router.models.items():
            supports = router.model_supports_reasoning(model_key)
            print(f"- {model_key}: {info['display_name']} (Reasoning: {'✓' if supports else '✗'}, Provider: {info['provider']})")
        
        print("\nReasoning max tokens:", router.config.get("reasoning_max_tokens", 2000))
    
    # Print model capabilities at startup
    debug_models()
    
    # Set up event handlers
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, model_dropdown, model_notification, reasoning_state, has_reasoning_state],
        outputs=[msg, chatbot, model_notification, reasoning_state, has_reasoning_state],
        queue=True
    )
    
    submit_btn.click(
        fn=respond,
        inputs=[msg, chatbot, model_dropdown, model_notification, reasoning_state, has_reasoning_state],
        outputs=[msg, chatbot, model_notification, reasoning_state, has_reasoning_state],
        queue=True
    )
    
    # Update clear function to also clear the reasoning
    clear_btn.click(
        fn=lambda: ([], "", "", False),
        outputs=[chatbot, model_notification, reasoning_state, has_reasoning_state],
        queue=False
    )