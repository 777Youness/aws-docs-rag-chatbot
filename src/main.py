import argparse

from src.rag import RAGSystem


def create_interface():
    import gradio as gr

    rag = RAGSystem()

    def respond(message, chat_history):
        try:
            bot_message = rag.generate(message)
        except Exception as e:
            bot_message = f"Error: {e}"
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    with gr.Blocks() as demo:
        gr.Markdown("# AWS Documentation RAG Chatbot")
        chatbot = gr.Chatbot(height=500, type="messages")
        msg = gr.Textbox(show_label=False, placeholder="Ask a question about AWS (S3, IAM, VPC)...", container=False)
        clear = gr.Button("Clear")

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: [], None, chatbot, queue=False)

    return demo


def run_cli():
    rag = RAGSystem()
    print("RAG system ready. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                break

            print("\nAssistant: ", end="", flush=True)
            print(rag.generate(query))

        except KeyboardInterrupt:
            print("\nSession ended.")
            break


def main():
    parser = argparse.ArgumentParser(description="AWS documentation RAG chatbot")
    parser.add_argument("--gradio", action="store_true", help="launch the Gradio web interface")
    args = parser.parse_args()

    if args.gradio:
        create_interface().launch()
    else:
        run_cli()


if __name__ == "__main__":
    main()
