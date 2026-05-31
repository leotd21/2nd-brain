"""
Chat interface for health Q&A.
"""

import logging
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from .rag_engine import RAGEngine

logger = logging.getLogger(__name__)
console = Console()


class ChatInterface:
    """
    Interactive chat interface for health questions.
    
    Example:
        chat = ChatInterface()
        chat.run()
    """
    
    WELCOME_MESSAGE = """
# 🧠 Second Brain - Trợ lý Sức khỏe

Xin chào! Tôi là trợ lý sức khỏe cá nhân của bạn.
Hãy đặt câu hỏi về sức khỏe, tôi sẽ trả lời dựa trên kiến thức từ các video y tế.

**Lệnh:**
- `quit` hoặc `exit`: Thoát
- `sources`: Xem nguồn tham khảo
- `help`: Hiển thị trợ giúp

⚠️ *Lưu ý: Đây chỉ là thông tin tham khảo, không thay thế tư vấn y tế.*
"""

    def __init__(self, rag_engine: RAGEngine = None):
        """Initialize chat interface."""
        self.rag_engine = rag_engine or RAGEngine()
        self.last_sources = []
        
    def run(self):
        """Run interactive chat loop."""
        console.print(Markdown(self.WELCOME_MESSAGE))
        console.print()
        
        while True:
            try:
                user_input = console.input("[bold blue]Bạn:[/] ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ("quit", "exit", "q"):
                    console.print("[yellow]Tạm biệt![/]")
                    break
                    
                if user_input.lower() == "help":
                    console.print(Markdown(self.WELCOME_MESSAGE))
                    continue
                    
                if user_input.lower() == "sources":
                    self._show_sources()
                    continue
                
                # Query RAG engine
                self._process_query(user_input)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Tạm biệt![/]")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                console.print(f"[red]Lỗi: {e}[/]")
    
    def _process_query(self, question: str):
        """Process a user question."""
        with console.status("[bold green]Đang tìm kiếm..."):
            response = self.rag_engine.query(question)
        
        self.last_sources = response.sources
        
        # Display answer
        console.print()
        console.print(Panel(
            Markdown(response.answer),
            title="[bold green]Trả lời[/]",
            border_style="green",
        ))
        
        # Show sources summary
        if response.sources:
            console.print(f"\n[dim]📚 {len(response.sources)} nguồn tham khảo[/]")
            console.print("[dim]Gõ 'sources' để xem chi tiết[/]")
        
        console.print()
    
    def _show_sources(self):
        """Display last query sources."""
        if not self.last_sources:
            console.print("[yellow]Chưa có nguồn tham khảo.[/]")
            return
        
        console.print("\n[bold]📚 Nguồn tham khảo:[/]\n")
        for i, source in enumerate(self.last_sources, 1):
            console.print(f"[cyan]{i}. {source.title}[/]")
            console.print(f"   [dim]{source.url}[/]")
            console.print(f"   [dim]Độ liên quan: {source.relevance_score:.2f}[/]")
            console.print()
