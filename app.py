import gradio as gr
# Gradio interface setup
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Video Dubbing")
    gr.Markdown("This tool uses AI to dub videos into different languages. Upload a video, choose a target language, and get a dubbed version!")
    
    with gr.Row():
        with gr.Column(scale=2):
                video = gr.Video(label="Upload Video (Optional)")
                video = gr.Textbox(label="YouTube URL (Optional)", placeholder="Enter YouTube URL")
                source_language = gr.Dropdown(
                    choices=["Spanish", "English", "French"],  # You can use `language_mapping.keys()` here
                    label="Target Language for Dubbing",
                    value="Spanish"
                )
                target_language = gr.Dropdown(
                    choices=["Spanish", "English", "French"],  # You can use `language_mapping.keys()` here
                    label="Target Language for Dubbing",
                    value="Spanish"
                )
                use_wav2lip = gr.Checkbox(
                    label="Use Wav2Lip for lip sync",
                    value=False,
                    info="Enable this if the video has close-up faces. May not work for all videos."
                )
                whisper_model = gr.Dropdown(
                    choices=["tiny", "base", "small", "medium", "large", "turbo"],
                    label="Whisper Model",
                    value="turbo"
                )
                bg_sound = gr.Checkbox(
                    label="Keep Background Sound",
                    value=False,
                    info="Keep background sound of the original video, may introduce noise."
                )
                submit_button = gr.Button("Process Video", variant="primary")
        
        with gr.Column(scale=2):
            output_video = gr.Video(label="Processed Video")
            error_message = gr.Textbox(label="Status/Error Message")

    submit_button.click(
        process_video, 
        inputs=[video, source_language, target_language, use_wav2lip, whisper_model, bg_sound], 
        outputs=[output_video, error_message]
    )

    gr.Markdown("""
    ## Notes:
    -
    
    """)

  

print("Launching Gradio interface...")
demo.queue()
demo.launch()
