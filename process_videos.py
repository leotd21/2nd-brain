"""
Process video transcripts and generate summaries.

Usage:
    1. Add your OpenAI API key to .env file
    2. Run: python process_videos.py
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.prompt import Confirm

# Load environment variables
load_dotenv()

console = Console()

def check_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        console.print(Panel(
            "[bold red]OpenAI API Key not found![/]\n\n"
            "Please add your API key to the .env file:\n"
            "[cyan]OPENAI_API_KEY=sk-your-key-here[/]\n\n"
            "Get your key at: https://platform.openai.com/api-keys",
            title="⚠️ Configuration Required"
        ))
        return False
    return True


def load_metadata():
    """Load video metadata."""
    metadata_file = Path("data/metadata/dr-tran-van-phuc.json")
    if not metadata_file.exists():
        console.print("[red]Metadata file not found. Run crawler first.[/]")
        return {}
    
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


def summarize_with_openai(transcript: str, title: str, channel: str) -> dict:
    """Generate summary using OpenAI API."""
    from openai import OpenAI
    
    client = OpenAI()
    
    # Truncate if too long
    max_chars = 15000
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "... [truncated]"
    
    prompt = f"""Bạn là một trợ lý y tế chuyên tóm tắt nội dung video sức khỏe bằng tiếng Việt.

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
    "summary": "Tóm tắt nội dung chính (2-3 đoạn văn, khoảng 200-300 từ)",
    "key_points": ["các điểm quan trọng (5-10 điểm)"],
    "health_advice": ["lời khuyên sức khỏe cụ thể"],
    "warnings": ["cảnh báo quan trọng nếu có"],
    "related_conditions": ["các bệnh/tình trạng sức khỏe liên quan"]
}}

Lưu ý:
- Giữ nguyên tiếng Việt
- Tập trung vào thông tin y tế hữu ích
- Không đưa ra chẩn đoán hoặc kê đơn thuốc
- Ghi chú nếu cần tham khảo bác sĩ"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful medical content summarizer. Always respond with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON from response
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end]
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end]
        
        return json.loads(content.strip())
        
    except json.JSONDecodeError:
        return {"summary": content, "error": "Failed to parse JSON"}
    except Exception as e:
        return {"error": str(e)}


def categorize(text: str, title: str) -> list:
    """Simple keyword-based categorization."""
    from src.processor.categorizer import Categorizer
    categorizer = Categorizer()
    return categorizer.categorize(text, title=title)


def save_summary(video_id: str, title: str, url: str, summary_data: dict, categories: list):
    """Save summary to files."""
    output_dir = Path("data/summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare full summary
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
    
    # Save Markdown for wiki
    md_path = output_dir / f"{video_id}.md"
    md_content = generate_markdown(full_summary)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return json_path, md_path


def generate_markdown(summary: dict) -> str:
    """Generate Obsidian-compatible markdown."""
    lines = [
        "---",
        f'title: "{summary.get("title", "Untitled")}"',
        f'video_id: {summary.get("video_id", "")}',
        f'source: {summary.get("source_url", "")}',
        f'processed: {summary.get("processed_at", "")[:10]}',
        f'tags: [{", ".join(summary.get("categories", []))}]',
        "---",
        "",
        f'# {summary.get("title", "Untitled")}',
        "",
        f'**Nguồn**: [{summary.get("source_url", "")}]({summary.get("source_url", "")})',
        "",
    ]
    
    # Topics
    if summary.get("main_topics"):
        topics = ", ".join(f"[[{t}]]" for t in summary["main_topics"])
        lines.extend(["## Chủ đề chính", topics, ""])
    
    # Summary
    if summary.get("summary"):
        lines.extend(["## Tóm tắt", summary["summary"], ""])
    
    # Key points
    if summary.get("key_points"):
        lines.append("## Điểm quan trọng")
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


def main():
    console.print(Panel.fit(
        "[bold blue]🧠 Second Brain - Processing Videos[/]\n"
        "Generating summaries with GPT-4o-mini",
        title="LLM Processing"
    ))
    
    # Check API key
    if not check_api_key():
        sys.exit(1)
    
    # Load metadata
    metadata = load_metadata()
    if not metadata:
        sys.exit(1)
    
    console.print(f"[green]✓[/] Loaded metadata for {len(metadata)} videos")
    
    # Find transcripts to process
    transcripts_dir = Path("data/transcripts")
    summaries_dir = Path("data/summaries")
    summaries_dir.mkdir(exist_ok=True)
    
    existing_summaries = {f.stem for f in summaries_dir.glob("*.json")}
    transcript_files = list(transcripts_dir.glob("*.json"))
    
    to_process = [f for f in transcript_files if f.stem not in existing_summaries]
    
    console.print(f"[green]✓[/] Found {len(transcript_files)} transcripts")
    console.print(f"[green]✓[/] Already processed: {len(existing_summaries)}")
    console.print(f"[yellow]→[/] Need to process: {len(to_process)}")
    
    if not to_process:
        console.print("\n[green]All videos already processed![/]")
        return
    
    # Estimate cost
    total_chars = sum(
        len(json.load(open(f, encoding="utf-8")).get("full_text", ""))
        for f in to_process
    )
    estimated_tokens = total_chars // 4
    estimated_cost = (estimated_tokens / 1000000) * 0.15  # GPT-4o-mini input price
    
    console.print(f"\n[dim]Estimated input: ~{estimated_tokens:,} tokens[/]")
    console.print(f"[dim]Estimated cost: ~${estimated_cost:.2f} USD[/]")
    
    if not Confirm.ask("\nProceed with processing?"):
        console.print("[yellow]Cancelled.[/]")
        return
    
    # Process videos
    console.print("\n[bold cyan]Processing videos...[/]\n")
    
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
        
        for transcript_file in to_process:
            video_id = transcript_file.stem
            video_meta = metadata.get(video_id, {})
            title = video_meta.get("title", video_id)
            
            progress.update(task, description=f"[cyan]{title[:35]}...[/]")
            
            try:
                # Load transcript
                transcript_data = load_transcript(video_id)
                if not transcript_data:
                    failed.append((video_id, "No transcript"))
                    progress.advance(task)
                    continue
                
                full_text = transcript_data.get("full_text", "")
                
                # Generate summary
                summary_data = summarize_with_openai(
                    transcript=full_text,
                    title=title,
                    channel="Dr. Trần Văn Phúc"
                )
                
                if "error" in summary_data and not summary_data.get("summary"):
                    failed.append((video_id, summary_data["error"]))
                    progress.advance(task)
                    continue
                
                # Categorize
                categories = categorize(
                    summary_data.get("summary", "") + " ".join(summary_data.get("key_points", [])),
                    title=title
                )
                
                # Save
                save_summary(
                    video_id=video_id,
                    title=title,
                    url=video_meta.get("url", f"https://youtube.com/watch?v={video_id}"),
                    summary_data=summary_data,
                    categories=categories
                )
                
                success += 1
                
            except Exception as e:
                failed.append((video_id, str(e)))
            
            progress.advance(task)
    
    # Summary
    console.print("\n" + "=" * 60)
    console.print(Panel.fit(
        f"[bold green]✓ Processing Complete![/]\n\n"
        f"Successfully processed: [bold]{success}[/]\n"
        f"Failed: [bold]{len(failed)}[/]\n"
        f"Total summaries: [bold]{len(list(summaries_dir.glob('*.json')))}[/]",
        title="Summary"
    ))
    
    if failed:
        console.print("\n[yellow]Failed videos:[/]")
        for vid_id, reason in failed[:5]:
            console.print(f"  [dim]- {vid_id}: {reason[:50]}[/]")
    
    console.print("\n[bold]Next steps:[/]")
    console.print("  1. Browse summaries in: [cyan]data/summaries/[/]")
    console.print("  2. Start chat: [cyan]python -m src.app.main[/]")


if __name__ == "__main__":
    main()
