"""
RAG (Retrieval Augmented Generation) engine for health Q&A.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from ..storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """A source document for RAG response."""
    video_id: str
    title: str
    url: str
    relevance_score: float
    snippet: str = ""


@dataclass
class RAGResponse:
    """Response from RAG query."""
    answer: str
    sources: List[Source] = field(default_factory=list)
    confidence: float = 0.0
    related_topics: List[str] = field(default_factory=list)


class RAGEngine:
    """
    RAG engine for answering health questions.
    
    Retrieves relevant content from vector store and generates
    answers using LLM with source citations.
    
    Example:
        engine = RAGEngine()
        response = engine.query("Vitamin D có tác dụng gì?")
        print(response.answer)
        for source in response.sources:
            print(f"- {source.title}: {source.url}")
    """
    
    SYSTEM_PROMPT = """Bạn là trợ lý sức khỏe cá nhân, trả lời câu hỏi dựa trên kiến thức từ các video y tế đáng tin cậy.

Quy tắc:
1. Chỉ trả lời dựa trên thông tin được cung cấp trong context
2. Nếu không có đủ thông tin, hãy nói rõ
3. Luôn khuyên người dùng tham khảo bác sĩ cho các vấn đề y tế nghiêm trọng
4. Trả lời bằng tiếng Việt
5. Trích dẫn nguồn khi có thể

⚠️ Lưu ý: Đây chỉ là thông tin tham khảo, không thay thế tư vấn y tế chuyên nghiệp."""

    QUERY_PROMPT = """Context từ các video y tế:
{context}

Câu hỏi: {question}

Hãy trả lời câu hỏi dựa trên context trên. Nếu context không đủ thông tin, hãy nói rõ."""

    def __init__(
        self,
        vector_store: VectorStore = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        top_k: int = 5,
    ):
        """
        Initialize RAG engine.
        
        Args:
            vector_store: Vector store instance
            llm_provider: LLM provider ("openai" or "anthropic")
            llm_model: Model name
            top_k: Number of documents to retrieve
        """
        self.vector_store = vector_store or VectorStore()
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.top_k = top_k
        self._llm_client = None
        
    def _get_llm_client(self):
        """Get or create LLM client."""
        if self._llm_client is not None:
            return self._llm_client
            
        if self.llm_provider == "openai":
            from openai import OpenAI
            self._llm_client = OpenAI()
        elif self.llm_provider == "anthropic":
            from anthropic import Anthropic
            self._llm_client = Anthropic()
            
        return self._llm_client
    
    def query(self, question: str) -> RAGResponse:
        """
        Answer a health question using RAG.
        
        Args:
            question: User's question
            
        Returns:
            RAGResponse with answer and sources
        """
        # Retrieve relevant documents
        results = self.vector_store.search(question, top_k=self.top_k)
        
        if not results:
            return RAGResponse(
                answer="Xin lỗi, tôi không tìm thấy thông tin liên quan trong cơ sở kiến thức.",
                confidence=0.0,
            )
        
        # Build context from results
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            metadata = result.get("metadata", {})
            context_parts.append(
                f"[{i+1}] {metadata.get('title', 'Unknown')}:\n{result['text'][:500]}"
            )
            sources.append(Source(
                video_id=metadata.get("video_id", result["id"]),
                title=metadata.get("title", "Unknown"),
                url=metadata.get("url", ""),
                relevance_score=result["score"],
                snippet=result["text"][:200],
            ))
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = self.QUERY_PROMPT.format(context=context, question=question)
        answer = self._generate_answer(prompt)
        
        # Calculate confidence based on relevance scores
        avg_score = sum(s.relevance_score for s in sources) / len(sources)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=avg_score,
        )
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using LLM."""
        client = self._get_llm_client()
        
        if self.llm_provider == "openai":
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            return response.choices[0].message.content
            
        elif self.llm_provider == "anthropic":
            response = client.messages.create(
                model=self.llm_model,
                max_tokens=1000,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
            
        return "Error: Unsupported LLM provider"
    
    def get_sources(self, question: str) -> List[Source]:
        """Get relevant sources without generating answer."""
        results = self.vector_store.search(question, top_k=self.top_k)
        
        sources = []
        for result in results:
            metadata = result.get("metadata", {})
            sources.append(Source(
                video_id=metadata.get("video_id", result["id"]),
                title=metadata.get("title", "Unknown"),
                url=metadata.get("url", ""),
                relevance_score=result["score"],
                snippet=result["text"][:200],
            ))
        
        return sources
