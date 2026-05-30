import os
import gradio as gr
from ingest import load_and_chunk
from retriever import build_vectorstore, load_vectorstore, retrieve
from llm import ask_gemini

vectorstore = None


def upload_pdf(pdf_file):
    global vectorstore
    if pdf_file is None:
        return "No file uploaded."

    chunks = load_and_chunk(pdf_file.name)
    vectorstore = build_vectorstore(chunks)
    return f"Done. {len(chunks)} chunks indexed from the document."


def answer_question(question):
    global vectorstore
    if vectorstore is None:
        return "Please upload a PDF first."
    if not question.strip():
        return "Please enter a question."

    results = retrieve(vectorstore, question, k=5)
    answer = ask_gemini(results, question)
    return answer


with gr.Blocks(title="Research Paper QA") as app:
    gr.Markdown("## Research Paper Question Answering")
    gr.Markdown("Upload a research paper and ask questions about its content.")

    with gr.Row():
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_status = gr.Textbox(label="Status", interactive=False)

    upload_btn = gr.Button("Process PDF")
    upload_btn.click(fn=upload_pdf, inputs=pdf_input, outputs=upload_status)

    gr.Markdown("---")

    question_input = gr.Textbox(label="Your Question", placeholder="What is the main contribution of this paper?")
    answer_output = gr.Textbox(label="Answer", lines=6, interactive=False)

    ask_btn = gr.Button("Ask")
    ask_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)


if __name__ == "__main__":
    app.launch()