"""
LLM-powered summarizer for health content.

Generates structured summaries from video transcripts.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


@dataclass
class VideoSummary:
    """Structured summary of a health video."""
    
    video_id: str
    title: str
    source_url: str
    main_topics: List[str] = field(default_factory=list)
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    health_advice: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    related_conditions: List[str] = field(default_factory=list)
    timestamps: Dict[str, str] = field(default_factory=dict)
    categories: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "source_url": self.source_url,
            "main_topics": self.main_topics,
            "summary": self.summary,
            "key_points": self.key_points,
            "health_advice": self.health_advice,
            "warnings": self.warnings,
            "related_conditions": self.related_conditions,
            "timestamps": self.timestamps,
            "categories": self.categories,
        }
    
    def to_markdown(self) -> str:
        """Convert summary to Obsidian-compatible markdown."""
        lines = [
            f"# {self.title}",
            "",
            f"**Source**: [{self.source_url}]({self.source_url})",
            f"**Video ID**: {self.video_id}",
            "",
            "## Topics",
            ", ".join(f"[[{topic}]]" for topic in self.main_topics),
            "",
            "## Summary",
            self.summary,
            "",
        ]
        
        if self.key_points:
            lines.extend([
                "## Key Points",
                *[f"- {point}" for point in self.key_points],
                "",
            ])
        
        if self.health_advice:
            lines.extend([
                "## Health Advice",
                *[f"- {advice}" for advice in self.health_advice],
                "",
            ])
        
        if self.warnings:
            lines.extend([
                "## ⚠️ Warnings",
                *[f"- {warning}" for warning in self.warnings],
                "",
            ])
        
        if self.related_conditions:
            lines.extend([
                "## Related Conditions",
                ", ".join(f"[[{cond}]]" for cond in self.related_conditions),
                "",
            ])
        
        if self.timestamps:
            lines.extend([
                "## Timestamps",
                *[f"- **{time}**: {topic}" for topic, time in self.timestamps.items()],
                "",
            ])
        
        if self.categories:
            lines.extend([
                "---",
                f"Tags: {', '.join(f'#{cat}' for cat in self.categories)}",
            ])
        
        return "\n".join(lines)


class Summarizer:
    """
    Generates structured summaries from video transcripts using LLM.
    
    Supports OpenAI and Anthropic APIs.
    
    Example:
        summarizer = Summarizer(provider="openai", model="gpt-4o-mini")
        summary = summarizer.summarize(transcript_text, video_metadata)
    """
    
    SUMMARY_PROMPT = """Bạn là một trợ lý y tế chuyên tóm tắt nội dung video sức khỏe bằng tiếng Việt.

Hãy phân tích transcript video sau và tạo một bản tóm tắt có cấu trúc.

**Transcript:**
{transcript}

**Thông tin video:**
- Tiêu đề: {title}
- Kênh: {channel}

**Yêu cầu:**
Trả về JSON với cấu trúc sau:
{{
    "main_topics": ["danh sách các chủ đề chính (3-5 chủ đề)"],
    "summary": "Tóm tắt nội dung chính (2-3 đoạn văn)",
    "key_points": ["các điểm quan trọng (5-10 điểm)"],
    "health_advice": ["lời khuyên sức khỏe cụ thể"],
    "warnings": ["cảnh báo quan trọng nếu có"],
    "related_conditions": ["các bệnh/tình trạng sức khỏe liên quan"],
    "timestamps": {{"chủ đề": "thời gian (nếu có thể xác định)"}}
}}

Lưu ý:
- Giữ nguyên tiếng Việt
- Tập trung vào thông tin y tế hữu ích
- Không đưa ra chẩn đoán hoặc kê đơn thuốc
- Ghi chú nếu cần tham khảo bác sĩ"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the summarizer.
        
        Args:
            provider: LLM provider ("openai" or "anthropic")
            model: Model name to use
            api_key: API key (uses environment variable if not provided)
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._client = None
        
    def _get_client(self):
        """Get or create LLM client."""
        if self._client is not None:
            return self._client
            
        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
            
        return self._client
    
    def summarize(
        self,
        transcript: str,
        title: str,
        channel: str,
        video_id: str,
        source_url: str,
    ) -> VideoSummary:
        """
        Generate a structured summary from transcript.
        
        Args:
            transcript: Full transcript text
            title: Video title
            channel: Channel name
            video_id: YouTube video ID
            source_url: Video URL
            
        Returns:
            VideoSummary object
        """
        # Truncate transcript if too long
        max_chars = 15000
        if len(transcript) > max_chars:
            transcript = transcript[:max_chars] + "... [truncated]"
        
        prompt = self.SUMMARY_PROMPT.format(
            transcript=transcript,
            title=title,
            channel=channel,
        )
        
        try:
            response_text = self._call_llm(prompt)
            summary_data = self._parse_response(response_text)
            
            return VideoSummary(
                video_id=video_id,
                title=title,
                source_url=source_url,
                main_topics=summary_data.get("main_topics", []),
                summary=summary_data.get("summary", ""),
                key_points=summary_data.get("key_points", []),
                health_advice=summary_data.get("health_advice", []),
                warnings=summary_data.get("warnings", []),
                related_conditions=summary_data.get("related_conditions", []),
                timestamps=summary_data.get("timestamps", {}),
            )
            
        except Exception as e:
            logger.error(f"Error summarizing video {video_id}: {e}")
            # Return minimal summary on error
            return VideoSummary(
                video_id=video_id,
                title=title,
                source_url=source_url,
                summary=f"Error generating summary: {str(e)}",
            )
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        client = self._get_client()
        
        if self.provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful medical content summarizer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content
            
        elif self.provider == "anthropic":
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return response.content[0].text
            
        raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _parse_response(self, response: str) -> dict:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response
        try:
            # Look for JSON block
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                response = response[start:end]
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                response = response[start:end]
            
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Return empty dict on parse error
            return {}
    
    def save_summary(
        self, 
        summary: VideoSummary, 
        output_dir: Path,
        format: str = "both"
    ) -> List[Path]:
        """
        Save summary to file(s).
        
        Args:
            summary: VideoSummary to save
            output_dir: Output directory
            format: "json", "markdown", or "both"
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        if format in ("json", "both"):
            json_path = output_dir / f"{summary.video_id}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
            saved_files.append(json_path)
            
        if format in ("markdown", "both"):
            md_path = output_dir / f"{summary.video_id}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(summary.to_markdown())
            saved_files.append(md_path)
            
        return saved_files
