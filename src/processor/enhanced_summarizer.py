"""
Enhanced Summarizer for Health Videos.

Improvements over basic summarization:
1. Multi-chunk processing for long transcripts
2. Medical-specific extraction (dosages, protocols, mechanisms)
3. Structured Q&A extraction
4. Timestamp-aware key moments
5. Better JSON validation and error handling
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from openai import OpenAI


@dataclass
class EnhancedSummary:
    """Structured summary with rich medical information."""
    video_id: str
    title: str
    source_url: str
    
    # Core content
    main_topics: list[str]
    summary: str
    detailed_summary: str  # NEW: Longer, more detailed summary
    
    # Medical specifics
    key_points: list[str]
    health_advice: list[str]
    warnings: list[str]
    related_conditions: list[str]
    
    # NEW: Enhanced fields
    mechanisms: list[str]  # How things work (e.g., "Cortisol ức chế hệ miễn dịch bằng cách...")
    specific_numbers: list[str]  # Dosages, percentages, durations
    protocols: list[str]  # Step-by-step recommendations
    myths_debunked: list[str]  # Common misconceptions addressed
    key_quotes: list[str]  # Important direct quotes from doctor
    
    # Metadata
    categories: list[str]
    processed_at: str
    quality_score: float  # 0-1 confidence score
    transcript_coverage: float  # % of transcript processed


class EnhancedSummarizer:
    """
    Multi-pass summarizer for comprehensive health video extraction.
    """
    
    def __init__(
        self,
        endpoint: str = None,
        model: str = None,
        chunk_size: int = 15000,  # Characters per chunk
        max_chunks: int = 8,  # Process up to 8 chunks (120k chars)
    ):
        endpoint = endpoint or os.environ.get("NINE_ROUTER_ENDPOINT", "https://9router.namnh.org/v1")
        model = model or os.environ.get("NINE_ROUTER_MODEL", "mrdev/kr/claude-opus-4.5")
        self.client = OpenAI(base_url=endpoint, api_key=os.environ.get("NINE_ROUTER_API_KEY", "not-needed"))
        self.model = model
        self.model = model
        self.chunk_size = chunk_size
        self.max_chunks = max_chunks
    
    def summarize(self, transcript: str, title: str, video_id: str) -> dict:
        """
        Generate comprehensive summary using multi-pass extraction.
        """
        # Split transcript into chunks
        chunks = self._split_transcript(transcript)
        total_chars = len(transcript)
        processed_chars = sum(len(c) for c in chunks)
        
        # Phase 1: Extract key information from each chunk
        chunk_extractions = []
        for i, chunk in enumerate(chunks):
            extraction = self._extract_from_chunk(chunk, title, i + 1, len(chunks))
            if extraction:
                chunk_extractions.append(extraction)
        
        # Phase 2: Synthesize all extractions into final summary
        final_summary = self._synthesize_summary(
            chunk_extractions, title, video_id, 
            coverage=processed_chars / total_chars if total_chars > 0 else 0
        )
        
        return final_summary
    
    def _split_transcript(self, transcript: str) -> list[str]:
        """Split transcript into processable chunks."""
        chunks = []
        
        # Try to split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', transcript)
        
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
                if len(chunks) >= self.max_chunks:
                    break
        
        if current_chunk and len(chunks) < self.max_chunks:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _extract_from_chunk(self, chunk: str, title: str, chunk_num: int, total_chunks: int) -> Optional[dict]:
        """Extract key information from a single chunk."""
        
        prompt = f"""Bạn là chuyên gia y tế phân tích video sức khỏe của Bác sĩ Trần Văn Phúc.

**Video:** {title}
**Phần:** {chunk_num}/{total_chunks}

**Nội dung:**
{chunk}

**Nhiệm vụ:** Trích xuất thông tin y tế quan trọng. Trả về JSON:

{{
    "topics": ["chủ đề chính trong phần này"],
    "key_facts": ["sự thật y khoa quan trọng, cụ thể"],
    "mechanisms": ["cơ chế hoạt động được giải thích (ví dụ: 'Cortisol làm tăng đường huyết bằng cách kích thích gan sản xuất glucose')"],
    "numbers": ["con số cụ thể: liều lượng, tỷ lệ, thời gian (ví dụ: 'Vitamin D: 1000-2000 IU/ngày', 'Nguy cơ tăng 40%')"],
    "advice": ["lời khuyên thực hành cụ thể"],
    "warnings": ["cảnh báo, chống chỉ định"],
    "conditions": ["bệnh lý được đề cập"],
    "myths": ["quan niệm sai được bác bỏ"],
    "quotes": ["câu nói đáng nhớ của bác sĩ (nguyên văn nếu có)"]
}}

**Quy tắc:**
- Chỉ trích xuất thông tin CÓ trong đoạn văn
- Ưu tiên thông tin CỤ THỂ, có thể hành động
- Giữ nguyên tiếng Việt
- Nếu không có thông tin cho mục nào, để mảng rỗng []
- Chỉ trả về JSON, không giải thích"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Trả về JSON hợp lệ. Không markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            
            content = response.choices[0].message.content
            return self._parse_json(content)
            
        except Exception as e:
            print(f"  Chunk {chunk_num} error: {e}")
            return None
    
    def _synthesize_summary(
        self, 
        extractions: list[dict], 
        title: str, 
        video_id: str,
        coverage: float
    ) -> dict:
        """Synthesize all chunk extractions into final comprehensive summary."""
        
        # Merge all extractions
        merged = {
            "topics": [],
            "key_facts": [],
            "mechanisms": [],
            "numbers": [],
            "advice": [],
            "warnings": [],
            "conditions": [],
            "myths": [],
            "quotes": []
        }
        
        for ext in extractions:
            if not ext:
                continue
            for key in merged:
                if key in ext and isinstance(ext[key], list):
                    merged[key].extend(ext[key])
        
        # Deduplicate
        for key in merged:
            merged[key] = list(dict.fromkeys(merged[key]))  # Preserve order, remove dupes
        
        # Generate final synthesis
        synthesis_prompt = f"""Bạn là chuyên gia y tế tổng hợp thông tin từ video sức khỏe.

**Video:** {title}

**Thông tin đã trích xuất:**
- Chủ đề: {json.dumps(merged['topics'][:10], ensure_ascii=False)}
- Sự thật y khoa: {json.dumps(merged['key_facts'][:15], ensure_ascii=False)}
- Cơ chế: {json.dumps(merged['mechanisms'][:10], ensure_ascii=False)}
- Con số cụ thể: {json.dumps(merged['numbers'][:10], ensure_ascii=False)}
- Lời khuyên: {json.dumps(merged['advice'][:10], ensure_ascii=False)}
- Cảnh báo: {json.dumps(merged['warnings'][:8], ensure_ascii=False)}
- Bệnh lý: {json.dumps(merged['conditions'][:10], ensure_ascii=False)}
- Quan niệm sai: {json.dumps(merged['myths'][:5], ensure_ascii=False)}
- Trích dẫn: {json.dumps(merged['quotes'][:5], ensure_ascii=False)}

**Nhiệm vụ:** Tổng hợp thành bản tóm tắt hoàn chỉnh. Trả về JSON:

{{
    "main_topics": ["3-5 chủ đề chính của toàn video"],
    "summary": "Tóm tắt ngắn 150-200 từ, nêu bật thông điệp chính",
    "detailed_summary": "Tóm tắt chi tiết 400-500 từ, bao gồm cơ chế và lý giải khoa học",
    "key_points": ["8-12 điểm quan trọng nhất, cụ thể và có thể hành động"],
    "health_advice": ["5-8 lời khuyên thực hành cụ thể"],
    "warnings": ["cảnh báo quan trọng"],
    "related_conditions": ["bệnh lý liên quan"],
    "mechanisms": ["3-5 cơ chế sinh học/y học quan trọng được giải thích"],
    "specific_numbers": ["liều lượng, tỷ lệ, thời gian cụ thể được đề cập"],
    "protocols": ["quy trình/phác đồ cụ thể nếu có"],
    "myths_debunked": ["quan niệm sai phổ biến được bác bỏ"],
    "key_quotes": ["1-3 câu nói đáng nhớ nhất"]
}}

**Quy tắc:**
- Tổng hợp KHÔNG lặp lại
- Ưu tiên thông tin CỤ THỂ, THỰC HÀNH được
- Giữ nguyên tiếng Việt
- Chỉ trả về JSON"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Trả về JSON hợp lệ. Không markdown."},
                    {"role": "user", "content": synthesis_prompt}
                ],
                temperature=0.3,
                max_tokens=3000,
            )
            
            content = response.choices[0].message.content
            result = self._parse_json(content)
            
            if result:
                # Add metadata
                result["video_id"] = video_id
                result["title"] = title
                result["source_url"] = f"https://www.youtube.com/watch?v={video_id}"
                result["processed_at"] = datetime.now().isoformat()
                result["transcript_coverage"] = round(coverage, 2)
                result["quality_score"] = self._calculate_quality_score(result)
                
                # Ensure all fields exist
                for field in ["mechanisms", "specific_numbers", "protocols", 
                             "myths_debunked", "key_quotes", "detailed_summary"]:
                    if field not in result:
                        result[field] = [] if field != "detailed_summary" else result.get("summary", "")
                
                return result
            
        except Exception as e:
            print(f"  Synthesis error: {e}")
        
        # Fallback: return merged data directly
        return {
            "video_id": video_id,
            "title": title,
            "source_url": f"https://www.youtube.com/watch?v={video_id}",
            "main_topics": merged["topics"][:5],
            "summary": "Tóm tắt không khả dụng",
            "detailed_summary": "",
            "key_points": merged["key_facts"][:10],
            "health_advice": merged["advice"][:8],
            "warnings": merged["warnings"][:5],
            "related_conditions": merged["conditions"][:8],
            "mechanisms": merged["mechanisms"][:5],
            "specific_numbers": merged["numbers"][:8],
            "protocols": [],
            "myths_debunked": merged["myths"][:5],
            "key_quotes": merged["quotes"][:3],
            "processed_at": datetime.now().isoformat(),
            "transcript_coverage": round(coverage, 2),
            "quality_score": 0.3
        }
    
    def _parse_json(self, content: str) -> Optional[dict]:
        """Robust JSON parsing with multiple fallback strategies."""
        
        # Strategy 1: Direct parse
        try:
            return json.loads(content)
        except:
            pass
        
        # Strategy 2: Extract from markdown code block
        if "```" in content:
            try:
                if "```json" in content:
                    json_str = content.split("```json")[1].split("```")[0]
                else:
                    json_str = content.split("```")[1].split("```")[0]
                return json.loads(json_str.strip())
            except:
                pass
        
        # Strategy 3: Find JSON object pattern
        try:
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                return json.loads(match.group())
        except:
            pass
        
        # Strategy 4: Fix common issues
        try:
            # Remove trailing commas
            fixed = re.sub(r',(\s*[}\]])', r'\1', content)
            # Fix unescaped quotes in strings
            fixed = re.sub(r'(?<!\\)"(?=[^"]*"[^"]*":)', '\\"', fixed)
            return json.loads(fixed)
        except:
            pass
        
        return None
    
    def _calculate_quality_score(self, summary: dict) -> float:
        """Calculate quality score based on completeness."""
        score = 0.0
        
        # Check required fields
        if summary.get("summary") and len(summary["summary"]) > 100:
            score += 0.2
        if summary.get("detailed_summary") and len(summary["detailed_summary"]) > 200:
            score += 0.1
        if len(summary.get("key_points", [])) >= 5:
            score += 0.2
        if len(summary.get("health_advice", [])) >= 3:
            score += 0.15
        if len(summary.get("mechanisms", [])) >= 1:
            score += 0.1
        if len(summary.get("specific_numbers", [])) >= 1:
            score += 0.1
        if summary.get("warnings"):
            score += 0.1
        if summary.get("related_conditions"):
            score += 0.05
        
        return min(1.0, score)


def save_enhanced_summary(summary: dict, output_dir: Path):
    """Save summary as both JSON and enhanced Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)
    video_id = summary["video_id"]
    
    # Save JSON
    with open(output_dir / f"{video_id}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Save enhanced Markdown
    md_content = generate_enhanced_markdown(summary)
    with open(output_dir / f"{video_id}.md", "w", encoding="utf-8") as f:
        f.write(md_content)


def generate_enhanced_markdown(s: dict) -> str:
    """Generate rich markdown from enhanced summary."""
    lines = [
        "---",
        f'title: "{s.get("title", "")}"',
        f'video_id: {s.get("video_id", "")}',
        f'source: {s.get("source_url", "")}',
        f'quality_score: {s.get("quality_score", 0):.2f}',
        f'coverage: {s.get("transcript_coverage", 0):.0%}',
        "---",
        "",
        f'# {s.get("title", "")}',
        "",
        f'📺 **Video**: [{s.get("source_url", "")}]({s.get("source_url", "")})',
        f'📊 **Chất lượng**: {s.get("quality_score", 0):.0%} | **Độ phủ**: {s.get("transcript_coverage", 0):.0%}',
        "",
    ]
    
    # Main topics as tags
    if s.get("main_topics"):
        topics = " • ".join(f"**{t}**" for t in s["main_topics"])
        lines.extend(["## 🏷️ Chủ đề chính", topics, ""])
    
    # Summary
    if s.get("summary"):
        lines.extend(["## 📝 Tóm tắt", s["summary"], ""])
    
    # Detailed summary (collapsible)
    if s.get("detailed_summary") and s["detailed_summary"] != s.get("summary"):
        lines.extend([
            "<details>",
            "<summary><b>📖 Xem tóm tắt chi tiết</b></summary>",
            "",
            s["detailed_summary"],
            "",
            "</details>",
            ""
        ])
    
    # Key points
    if s.get("key_points"):
        lines.append("## ✅ Điểm quan trọng")
        for p in s["key_points"]:
            lines.append(f"- {p}")
        lines.append("")
    
    # Mechanisms (scientific explanations)
    if s.get("mechanisms"):
        lines.append("## 🔬 Cơ chế khoa học")
        for m in s["mechanisms"]:
            lines.append(f"- {m}")
        lines.append("")
    
    # Specific numbers
    if s.get("specific_numbers"):
        lines.append("## 📊 Con số cụ thể")
        for n in s["specific_numbers"]:
            lines.append(f"- {n}")
        lines.append("")
    
    # Health advice
    if s.get("health_advice"):
        lines.append("## 💡 Lời khuyên thực hành")
        for a in s["health_advice"]:
            lines.append(f"- {a}")
        lines.append("")
    
    # Protocols
    if s.get("protocols"):
        lines.append("## 📋 Phác đồ/Quy trình")
        for i, p in enumerate(s["protocols"], 1):
            lines.append(f"{i}. {p}")
        lines.append("")
    
    # Warnings
    if s.get("warnings"):
        lines.append("## ⚠️ Cảnh báo")
        for w in s["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    
    # Myths debunked
    if s.get("myths_debunked"):
        lines.append("## 🚫 Quan niệm sai được bác bỏ")
        for m in s["myths_debunked"]:
            lines.append(f"- ❌ {m}")
        lines.append("")
    
    # Key quotes
    if s.get("key_quotes"):
        lines.append("## 💬 Trích dẫn đáng nhớ")
        for q in s["key_quotes"]:
            lines.append(f'> "{q}"')
        lines.append("")
    
    # Related conditions
    if s.get("related_conditions"):
        conditions = ", ".join(f"[[{c}]]" for c in s["related_conditions"])
        lines.extend(["## 🏥 Bệnh lý liên quan", conditions, ""])
    
    # Footer
    lines.extend([
        "---",
        f'*Xử lý: {s.get("processed_at", "")[:10]}*'
    ])
    
    return "\n".join(lines)
