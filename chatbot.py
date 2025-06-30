from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(temperature=0.5, model='gpt-4')

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
retriever = vector_store.as_retriever(search_kwargs={'k': 5})


# RAG function
def stream_response(message, history):
    docs = retriever.invoke(message)
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""
    You are an assistant which answers questions based on knowledge which is provided to you.
    While answering, you don't use your internal knowledge, 
    but solely the information in the "The knowledge" section.
    Do not mention the knowledge explicitly.

    The question: {message}

    Conversation history:
    {history}

    The knowledge:
    {knowledge}
    """

    partial_message = ""
    for response in llm.stream(rag_prompt):
        partial_message += response.content
        yield partial_message


# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 800px !important;
    margin: 0 auto !important;
}
#chatbot {
    min-height: 500px;
    border-radius: 10px;
}
.prompt-btn {
    margin: 5px;
    flex-grow: 1;
}
.prompt-row {
    margin-bottom: 15px;
}
.footer {
    font-size: 0.8em;
    color: #666;
    text-align: center;
    margin-top: 20px;
}
"""

# Interface with title and prompt buttons
with gr.Blocks(title="Document Knowledge Assistant", css=custom_css) as demo:
    # Header section
    gr.Markdown("""
    # 📚 Document Knowledge Assistant
    *Ask questions about the embedded documents using RAG technology*
    """)

    # Quick prompt buttons
    with gr.Row(variant="panel", elem_classes="prompt-row"):
        gr.Markdown("**Try these questions:**")

    with gr.Row(variant="panel", elem_classes="prompt-row"):
        prompt_buttons = []
        for prompt in [
            "What is this about?",
            "Key findings summary",
            "Main topics covered",
            "Important dates/timelines",
            "Technical terms explained"
        ]:
            btn = gr.Button(prompt, elem_classes="prompt-btn")
            prompt_buttons.append(btn)

    # Chat interface
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        show_copy_button=True,
        height=500,
        layout="panel"
    )

    # Input components
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your question about the documents...",
            container=False,
            autofocus=True,
            scale=7,
            min_width=600
        )
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    # Footer
    gr.Markdown("""
    <div class="footer">
    Powered by OpenAI, LangChain, and ChromaDB | Documents processed using text-embedding-3-large
    </div>
    """)


    # Event handlers
    def respond(message, chat_history):
        bot_message = ""
        for response in stream_response(message, chat_history):
            bot_message = response
        chat_history.append((message, bot_message))
        return "", chat_history


    def clear_chat():
        return []


    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_chat, None, chatbot)

    # Connect prompt buttons to chat
    for btn in prompt_buttons:
        btn.click(
            lambda x: x,
            inputs=[gr.State(btn.value)],
            outputs=[msg]
        ).then(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,
        favicon_path="https://uploads-ssl.webflow.com/6097e0eca1e87557da031fef/646b9ee3b4e8b937a5a8437f_ai%20brain%20icon%20light.svg"
    )