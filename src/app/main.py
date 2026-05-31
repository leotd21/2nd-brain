"""
Main entry point for the application.

Usage:
    python -m src.app.main [--mode cli|gradio]
"""

import argparse
import logging

from .chat import ChatInterface
from .rag_engine import RAGEngine
from ..storage.vector_store import VectorStore


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def run_cli():
    """Run CLI chat interface."""
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store)
    chat = ChatInterface(rag_engine=rag_engine)
    chat.run()


def run_gradio():
    """Run Gradio web interface."""
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        return
    
    vector_store = VectorStore()
    rag_engine = RAGEngine(vector_store=vector_store)
    
    def respond(message, history):
        response = rag_engine.query(message)
        sources_text = ""
        if response.sources:
            sources_text = "\n\n**Nguồn:**\n"
            for s in response.sources[:3]:
                sources_text += f"- [{s.title}]({s.url})\n"
        return response.answer + sources_text
    
    demo = gr.ChatInterface(
        respond,
        title="🧠 Second Brain - Trợ lý Sức khỏe",
        description="Đặt câu hỏi về sức khỏe",
        theme="soft",
    )
    
    demo.launch()


def main():
    parser = argparse.ArgumentParser(description="Health Knowledge Assistant")
    parser.add_argument("--mode", choices=["cli", "gradio"], default="cli")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.mode == "cli":
        run_cli()
    else:
        run_gradio()


if __name__ == "__main__":
    main()
