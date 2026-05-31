"""
Chat with your Second Brain - Health Knowledge Assistant.

Usage:
    python chat.py                    # Interactive chat
    python chat.py "your question"    # Single question
    python chat.py --web              # Web interface (Gradio)
"""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

# Configuration
CHROMA_PATH = "./data/chroma"
COLLECTION_NAME = "health_knowledge"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_ENDPOINT = "http://localhost:20128/v1"
LLM_MODEL = "kr/claude-opus-4.5"

# Initialize clients
llm_client = OpenAI(base_url=LOCAL_ENDPOINT, api_key="not-needed")


def get_collection():
    """Get ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )


def search(query: str, top_k: int = 5) -> list:
    """Search for relevant documents."""
    collection = get_collection()
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    sources = []
    for i in range(len(results["ids"][0])):
        sources.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i]  # Convert distance to similarity
        })
    
    return sources


def generate_answer(question: str, sources: list) -> str:
    """Generate answer using LLM with retrieved context."""
    
    # Build context from sources
    context_parts = []
    for i, src in enumerate(sources, 1):
        meta = src["metadata"]
        context_parts.append(
            f"[Nguồn {i}: {meta.get('title', 'Unknown')}]\n{src['text'][:800]}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Bạn là trợ lý sức khỏe cá nhân. Trả lời câu hỏi dựa trên kiến thức từ các video y tế của Bác sĩ Trần Văn Phúc.

**Quy tắc:**
1. Chỉ trả lời dựa trên thông tin trong context
2. Nếu không đủ thông tin, nói rõ
3. Luôn khuyên tham khảo bác sĩ cho vấn đề nghiêm trọng
4. Trả lời bằng tiếng Việt
5. Trích dẫn nguồn video khi có thể

**Context từ các video:**
{context}

**Câu hỏi:** {question}

**Trả lời:**"""

    response = llm_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Bạn là trợ lý sức khỏe thông minh, trả lời chính xác và hữu ích."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000,
    )
    
    return response.choices[0].message.content


def ask(question: str, show_sources: bool = True) -> tuple:
    """Ask a question and get an answer with sources."""
    
    # Search for relevant content
    sources = search(question, top_k=5)
    
    if not sources:
        return "Không tìm thấy thông tin liên quan trong cơ sở kiến thức.", []
    
    # Generate answer
    answer = generate_answer(question, sources)
    
    return answer, sources


def interactive_chat():
    """Run interactive chat session."""
    console.print(Panel.fit(
        "[bold blue]🧠 Second Brain - Trợ lý Sức khỏe[/]\n\n"
        "Hỏi bất kỳ câu hỏi nào về sức khỏe!\n"
        "Kiến thức từ 110 video của Bác sĩ Trần Văn Phúc.\n\n"
        "[dim]Gõ 'quit' để thoát, 'sources' để xem nguồn[/]",
        title="Xin chào!"
    ))
    
    last_sources = []
    
    while True:
        try:
            console.print()
            question = console.input("[bold cyan]Bạn:[/] ").strip()
            
            if not question:
                continue
            
            if question.lower() in ("quit", "exit", "q", "thoát"):
                console.print("[yellow]Tạm biệt! 👋[/]")
                break
            
            if question.lower() == "sources":
                if last_sources:
                    console.print("\n[bold]📚 Nguồn tham khảo:[/]")
                    for i, src in enumerate(last_sources, 1):
                        meta = src["metadata"]
                        console.print(f"\n[cyan]{i}. {meta.get('title', 'Unknown')}[/]")
                        console.print(f"   [dim]{meta.get('url', '')}[/]")
                        console.print(f"   [dim]Độ liên quan: {src['score']:.2f}[/]")
                else:
                    console.print("[dim]Chưa có nguồn tham khảo.[/]")
                continue
            
            # Get answer
            with console.status("[bold green]Đang tìm kiếm..."):
                answer, sources = ask(question)
            
            last_sources = sources
            
            # Display answer
            console.print()
            console.print(Panel(
                Markdown(answer),
                title="[bold green]Trả lời[/]",
                border_style="green"
            ))
            
            # Show source hint
            if sources:
                titles = [s["metadata"].get("title", "")[:30] for s in sources[:3]]
                console.print(f"\n[dim]📚 Nguồn: {', '.join(titles)}...[/]")
                console.print("[dim]Gõ 'sources' để xem chi tiết[/]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Tạm biệt! 👋[/]")
            break
        except Exception as e:
            console.print(f"[red]Lỗi: {e}[/]")


def run_gradio():
    """Run Gradio web interface."""
    try:
        import gradio as gr
    except ImportError:
        console.print("[red]Gradio not installed. Run: pip install gradio[/]")
        return
    
    def respond(message, history):
        answer, sources = ask(message)
        
        # Add sources to answer
        if sources:
            answer += "\n\n---\n**📚 Nguồn tham khảo:**\n"
            for i, src in enumerate(sources[:3], 1):
                meta = src["metadata"]
                answer += f"\n{i}. [{meta.get('title', 'Unknown')}]({meta.get('url', '')})"
        
        return answer
    
    demo = gr.ChatInterface(
        respond,
        title="🧠 Second Brain - Trợ lý Sức khỏe",
        description="Hỏi đáp về sức khỏe dựa trên kiến thức từ Bác sĩ Trần Văn Phúc",
        examples=[
            "Lectin là gì và tại sao nó có hại?",
            "Vitamin B12 có tác dụng gì?",
            "Làm thế nào để giảm viêm mãn tính?",
            "Cortisol ảnh hưởng đến sức khỏe như thế nào?",
        ],
        theme="soft",
    )
    
    console.print("[green]Starting web interface at http://localhost:7860[/]")
    demo.launch()


def main():
    parser = argparse.ArgumentParser(description="Chat with your health knowledge base")
    parser.add_argument("question", nargs="?", help="Question to ask (optional)")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--no-sources", action="store_true", help="Don't show sources")
    args = parser.parse_args()
    
    if args.web:
        run_gradio()
    elif args.question:
        # Single question mode
        answer, sources = ask(args.question)
        console.print(Panel(Markdown(answer), title="Trả lời", border_style="green"))
        
        if sources and not args.no_sources:
            console.print("\n[bold]📚 Nguồn:[/]")
            for src in sources[:3]:
                meta = src["metadata"]
                console.print(f"  - {meta.get('title', 'Unknown')}")
                console.print(f"    [dim]{meta.get('url', '')}[/]")
    else:
        # Interactive mode
        interactive_chat()


if __name__ == "__main__":
    main()
