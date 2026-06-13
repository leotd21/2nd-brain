"""
Process all video transcripts using local 9router LLM.
"""

import json
import os
import time
import sys
from pathlib import Path
from datetime import datetime
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

console = Console()

# Configure for local 9router
LOCAL_ENDPOINT = os.environ.get("NINE_ROUTER_ENDPOINT", "https://9router.namnh.org/v1")
MODEL = os.environ.get("NINE_ROUTER_MODEL", "mrdev/kr/claude-opus-4.5")

client = OpenAI(base_url=LOCAL_ENDPOINT, api_key=os.environ.get("NINE_ROUTER_API_KEY", "not-needed"))


def load_metadata():
    """Load video metadata."""
    metadata_file = Path("data/metadata/dr-tran-van-phuc.json")
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {v["id"]: v for v in data.get("videos", [])}


def load_transcript(video_id: str) -> dict:
    """Load transcript for a video."""
    transcript_file = Path(f"data/transcripts/{video_id}.json")
    if not transcript_file.exists():
        return None
    with open(transcript_file, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize(transcript: str, title: str) -> dict:
    """Generate summary using local LLM."""
    # Truncate if too long (keep ~12000 chars for context)
    max_chars = 12000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "... [truncated]"
    
    prompt = f"""Bạn là trợ lý y tế chuyên tóm tắt video sức khỏe tiếng Việt.

**Tiêu đề video:** {title}

**Transcript:**
{transcript}

**Yêu cầu:** Trả về JSON với cấu trúc:
{{
    "main_topics": ["3-5 chủ đề chính"],
    "summary": "Tóm tắt 200-300 từ",
    "key_points": ["5-10 điểm quan trọng"],
    "health_advice": ["lời khuyên sức khỏe"],
    "warnings": ["cảnh báo nếu có"],
    "related_conditions": ["bệnh lý liên quan"]
}}

Giữ nguyên tiếng Việt. Chỉ trả về JSON, không giải thích thêm."""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a medical content summarizer. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        
        content = response.choices[0].message.content
        
        # Extract JSON from response
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end]
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end]
        
        return json.loads(content.strip())
        
    except json.JSONDecodeError as e:
        return {"summary": content, "parse_error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def categorize(text: str, title: str) -> list:
    """Simple keyword-based categorization."""
    from src.processor.categorizer import Categorizer
    categorizer = Categorizer()
    return categorizer.categorize(text, title=title)


def save_summary(video_id: str, title: str, url: str, summary_data: dict, categories: list):
    """Save summary to JSON and Markdown."""
    output_dir = Path("data/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    full_summary = {
        "video_id": video_id,
        "title": title,
        "source_url": url,
        "categories": categories,
        "processed_at": datetime.now().isoformat(),
        **summary_data
    }
    
    # Save JSON
    json_path = output_dir / f"{video_id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, ensure_ascii=False, indent=2)
    
    # Save Markdown
    md_content = generate_markdown(full_summary)
    md_path = output_dir / f"{video_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def generate_markdown(s: dict) -> str:
    """Generate Obsidian-compatible markdown."""
    lines = [
        "---",
        f'title: "{s.get("title", "")}"',
        f'video_id: {s.get("video_id", "")}',
        f'source: {s.get("source_url", "")}',
        f'tags: [{", ".join(s.get("categories", []))}]',
        "---",
        "",
        f'# {s.get("title", "")}',
        "",
        f'**Nguồn**: [{s.get("source_url", "")}]({s.get("source_url", "")})',
        "",
    ]
    
    if s.get("main_topics"):
        lines.extend(["## Chủ đề", ", ".join(f"[[{t}]]" for t in s["main_topics"]), ""])
    
    if s.get("summary"):
        lines.extend(["## Tóm tắt", s["summary"], ""])
    
    if s.get("key_points"):
        lines.append("## Điểm quan trọng")
        for p in s["key_points"]:
            lines.append(f"- {p}")
        lines.append("")
    
    if s.get("health_advice"):
        lines.append("## Lời khuyên")
        for a in s["health_advice"]:
            lines.append(f"- {a}")
        lines.append("")
    
    if s.get("warnings"):
        lines.append("## ⚠️ Cảnh báo")
        for w in s["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    
    if s.get("related_conditions"):
        lines.extend(["## Bệnh lý liên quan", ", ".join(f"[[{c}]]" for c in s["related_conditions"]), ""])
    
    return "\n".join(lines)


def main():
    console.print(Panel.fit(
        f"[bold blue]🧠 Processing Videos with Local LLM[/]\n"
        f"Endpoint: {LOCAL_ENDPOINT}\n"
        f"Model: {MODEL}",
        title="Second Brain"
    ))
    
    # Load data
    metadata = load_metadata()
    console.print(f"[green]✓[/] Loaded {len(metadata)} video metadata")
    
    # Find what needs processing
    transcripts_dir = Path("data/transcripts")
    summaries_dir = Path("data/summaries")
    summaries_dir.mkdir(exist_ok=True)
    
    existing = {f.stem for f in summaries_dir.glob("*.json")}
    transcript_files = list(transcripts_dir.glob("*.json"))
    to_process = [f for f in transcript_files if f.stem not in existing]
    
    console.print(f"[green]✓[/] Transcripts: {len(transcript_files)}")
    console.print(f"[green]✓[/] Already done: {len(existing)}")
    console.print(f"[yellow]→[/] To process: {len(to_process)}")
    
    if not to_process:
        console.print("\n[green]All videos already processed![/]")
        return
    
    # Process
    success = 0
    failed = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=len(to_process))
        
        for tf in to_process:
            video_id = tf.stem
            meta = metadata.get(video_id, {})
            title = meta.get("title", video_id)
            
            progress.update(task, description=f"[cyan]{title[:30]}...[/]")
            
            try:
                # Load transcript
                tdata = load_transcript(video_id)
                if not tdata:
                    failed.append((video_id, "No transcript"))
                    progress.advance(task)
                    continue
                
                # Summarize
                summary = summarize(tdata.get("full_text", ""), title)
                
                if "error" in summary and not summary.get("summary"):
                    failed.append((video_id, summary.get("error", "Unknown")))
                    progress.advance(task)
                    continue
                
                # Categorize
                text_for_cat = summary.get("summary", "") + " ".join(summary.get("key_points", []))
                categories = categorize(text_for_cat, title)
                
                # Save
                url = meta.get("url", f"https://youtube.com/watch?v={video_id}")
                save_summary(video_id, title, url, summary, categories)
                success += 1
                
            except Exception as e:
                failed.append((video_id, str(e)[:50]))
            
            progress.advance(task)
            time.sleep(0.5)  # Small delay between requests
    
    # Summary
    total = len(list(summaries_dir.glob("*.json")))
    
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        f"[bold green]✓ Processing Complete![/]\n\n"
        f"Processed: [bold]{success}[/]\n"
        f"Failed: [bold]{len(failed)}[/]\n"
        f"Total summaries: [bold]{total}[/]",
        title="Summary"
    ))
    
    if failed:
        console.print("\n[yellow]Failed:[/]")
        for vid, reason in failed[:10]:
            console.print(f"  [dim]- {vid}: {reason}[/]")


if __name__ == "__main__":
    main()
