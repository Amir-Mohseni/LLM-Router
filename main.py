from app.app import demo

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        share=True,  # This creates a public shareable link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860  # Use standard Gradio port
    ) 