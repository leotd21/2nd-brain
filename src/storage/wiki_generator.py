"""
Wiki generator for creating Obsidian-compatible markdown files.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


class WikiGenerator:
    """
    Generates Obsidian-compatible markdown wiki from summaries.
    
    Creates a structured knowledge base with:
    - Topic-based organization
    - Backlinks and tags
    - Index pages
    
    Example:
        generator = WikiGenerator(output_dir="./knowledge_base")
        generator.generate_from_summaries(summaries)
    """
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize the wiki generator.
        
        Args:
            output_dir: Root directory for the wiki
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./knowledge_base")
        self._setup_directories()
        
    def _setup_directories(self):
        """Create wiki directory structure."""
        dirs = [
            self.output_dir,
            self.output_dir / "topics",
            self.output_dir / "sources",
            self.output_dir / "tags",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def generate_video_page(self, summary: Dict, source_name: str = "dr-tran-van-phuc") -> Path:
        """
        Generate a wiki page for a video summary.
        
        Args:
            summary: Video summary dict
            source_name: Source identifier
            
        Returns:
            Path to generated file
        """
        video_id = summary.get("video_id", "unknown")
        title = summary.get("title", "Untitled")
        
        # Create source directory
        source_dir = self.output_dir / "sources" / source_name / "videos"
        source_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate markdown content
        content = self._format_video_page(summary)
        
        # Save file
        file_path = source_dir / f"{video_id}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Generated wiki page: {file_path}")
        return file_path
    
    def _format_video_page(self, summary: Dict) -> str:
        """Format summary as Obsidian markdown."""
        lines = [
            "---",
            f"title: \"{summary.get('title', 'Untitled')}\"",
            f"video_id: {summary.get('video_id', '')}",
            f"source: {summary.get('source_url', '')}",
            f"created: {datetime.now().strftime('%Y-%m-%d')}",
            f"tags: [{', '.join(summary.get('categories', []))}]",
            "---",
            "",
            f"# {summary.get('title', 'Untitled')}",
            "",
            f"**Nguồn**: [{summary.get('source_url', '')}]({summary.get('source_url', '')})",
            "",
        ]
        
        # Topics with wiki links
        if summary.get("main_topics"):
            topics = ", ".join(f"[[{t}]]" for t in summary["main_topics"])
            lines.extend(["## Chủ đề", topics, ""])
        
        # Summary
        if summary.get("summary"):
            lines.extend(["## Tóm tắt", summary["summary"], ""])
        
        # Key points
        if summary.get("key_points"):
            lines.append("## Điểm chính")
            for point in summary["key_points"]:
                lines.append(f"- {point}")
            lines.append("")
        
        # Health advice
        if summary.get("health_advice"):
            lines.append("## Lời khuyên sức khỏe")
            for advice in summary["health_advice"]:
                lines.append(f"- {advice}")
            lines.append("")
        
        # Warnings
        if summary.get("warnings"):
            lines.append("## ⚠️ Cảnh báo")
            for warning in summary["warnings"]:
                lines.append(f"- {warning}")
            lines.append("")
        
        # Related conditions
        if summary.get("related_conditions"):
            conditions = ", ".join(f"[[{c}]]" for c in summary["related_conditions"])
            lines.extend(["## Bệnh lý liên quan", conditions, ""])
        
        return "\n".join(lines)
    
    def generate_topic_index(self, topic: str, video_ids: List[str]) -> Path:
        """Generate an index page for a topic."""
        topic_dir = self.output_dir / "topics"
        topic_dir.mkdir(exist_ok=True)
        
        content = [
            f"# {topic}",
            "",
            "## Videos",
            "",
        ]
        
        for vid in video_ids:
            content.append(f"- [[{vid}]]")
        
        file_path = topic_dir / f"{topic.lower().replace(' ', '-')}.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        return file_path
    
    def generate_main_index(self, stats: Dict = None) -> Path:
        """Generate the main index page."""
        content = [
            "# 🧠 Second Brain - Kiến thức Sức khỏe",
            "",
            "Cơ sở kiến thức sức khỏe cá nhân được xây dựng từ các nguồn y tế đáng tin cậy.",
            "",
            "## 📚 Chủ đề",
            "",
            "- [[Dinh dưỡng]]",
            "- [[Bệnh lý]]",
            "- [[Lối sống]]",
            "- [[Thuốc]]",
            "- [[Sức khỏe tâm thần]]",
            "- [[Phòng bệnh]]",
            "",
            "## 📺 Nguồn",
            "",
            "- [[Dr. Trần Văn Phúc|sources/dr-tran-van-phuc/_index]]",
            "",
            "## ⚠️ Lưu ý",
            "",
            "Đây là công cụ tham khảo cá nhân, KHÔNG thay thế tư vấn y tế chuyên nghiệp.",
            "",
            "---",
            f"*Cập nhật: {datetime.now().strftime('%Y-%m-%d')}*",
        ]
        
        file_path = self.output_dir / "index.md"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))
        
        return file_path
