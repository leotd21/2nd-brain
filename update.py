"""
Update Second Brain with new videos from the channel.

Usage:
    python update.py              # Check for new videos and process them
    python update.py --force      # Re-process all videos
    python update.py --check      # Just check for new videos, don't process
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.crawler.youtube_scraper import YouTubeScraper
from src.crawler.transcript_extractor import TranscriptExtractor
from src.processor.categorizer import Categorizer

console = Console()

# Configuration
CHANNEL_ID = "@BacsiTranVanPhucOfficial"
LOCAL_ENDPOINT = os.environ.get("NINE_ROUTER_ENDPOINT", "https://9router.namnh.org/v1")
LLM_MODEL = os.environ.get("NINE_ROUTER_MODEL", "mrdev/kr/claude-opus-4.5")

client = OpenAI(base_url=LOCAL_ENDPOINT, api_key=os.environ.get("NINE_ROUTER_API_KEY", "not-needed"))


def crawl_new_videos():
    """Crawl channel and return new videos."""
    console.print("\n[bold cyan]Step 1: Checking for new videos...[/]")
    
    data_dir = Path("./data")
    scraper = YouTubeScraper(output_dir=data_dir / "metadata")
    
    # Get all videos from channel
    with console.status("[bold green]Fetching video list..."):
        videos = scraper.get_channel_videos(CHANNEL_ID, limit=None)
    
    # Check which ones we already have transcripts for
    existing_transcripts = {f.stem for f in (data_dir / "transcripts").glob("*.json")}
    new_videos = [v for v in videos if v.id not in existing_transcripts]
    
    console.print(f"  Total videos on channel: {len(videos)}")
    console.print(f"  Already have: {len(existing_transcripts)}")
    console.print(f"  [yellow]New videos: {len(new_videos)}[/]")
    
    # Save updated metadata
    scraper.save_metadata(videos, "dr-tran-van-phuc")
    
    return new_videos, videos


def extract_transcripts(videos):
    """Extract transcripts for new videos."""
    if not videos:
        return []
    
    console.print(f"\n[bold cyan]Step 2: Extracting {len(videos)} transcripts...[/]")
    
    extractor = TranscriptExtractor(
        output_dir=Path("./data/transcripts"),
        use_whisper_fallback=False
    )
    
    extracted = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Extracting...", total=len(videos))
        
        for video in videos:
            progress.update(task, description=f"[cyan]{video.title[:30]}...[/]")
            
            try:
                transcript = extractor.get_transcript(video.id)
                if transcript:
                    extractor.save_transcript(transcript)
                    extracted.append(video.id)
            except Exception as e:
                console.print(f"  [dim]Skip {video.id}: {str(e)[:30]}[/]")
            
            progress.advance(task)
            time.sleep(0.3)
    
    console.print(f"  [green]Extracted: {len(extracted)} transcripts[/]")
    return extracted


def summarize_video(transcript: str, title: str) -> dict:
    """Generate summary using local LLM."""
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

Giữ nguyên tiếng Việt. Chỉ trả về JSON."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=2000,
    )
    
    content = response.choices[0].message.content
    
    # Extract JSON
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    return json.loads(content.strip())


def process_new_summaries(video_ids):
    """Generate summaries for new videos."""
    if not video_ids:
        return
    
    console.print(f"\n[bold cyan]Step 3: Generating {len(video_ids)} summaries...[/]")
    
    # Load metadata
    with open("data/metadata/dr-tran-van-phuc.json", "r", encoding="utf-8") as f:
        metadata = {v["id"]: v for v in json.load(f).get("videos", [])}
    
    categorizer = Categorizer()
    summaries_dir = Path("data/summaries")
    summaries_dir.mkdir(exist_ok=True)
    
    success = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Summarizing...", total=len(video_ids))
        
        for video_id in video_ids:
            meta = metadata.get(video_id, {})
            title = meta.get("title", video_id)
            
            progress.update(task, description=f"[cyan]{title[:30]}...[/]")
            
            try:
                # Load transcript
                with open(f"data/transcripts/{video_id}.json", "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)
                
                # Summarize
                summary = summarize_video(transcript_data.get("full_text", ""), title)
                
                # Categorize
                text = summary.get("summary", "") + " ".join(summary.get("key_points", []))
                categories = categorizer.categorize(text, title=title)
                
                # Save
                full_summary = {
                    "video_id": video_id,
                    "title": title,
                    "source_url": meta.get("url", f"https://youtube.com/watch?v={video_id}"),
                    "categories": categories,
                    "processed_at": datetime.now().isoformat(),
                    **summary
                }
                
                with open(summaries_dir / f"{video_id}.json", "w", encoding="utf-8") as f:
                    json.dump(full_summary, f, ensure_ascii=False, indent=2)
                
                # Also save markdown
                save_markdown(full_summary, summaries_dir / f"{video_id}.md")
                
                success += 1
                
            except Exception as e:
                console.print(f"  [red]Error {video_id}: {str(e)[:40]}[/]")
            
            progress.advance(task)
            time.sleep(0.5)
    
    console.print(f"  [green]Summarized: {success} videos[/]")


def save_markdown(s: dict, path: Path):
    """Save summary as markdown."""
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
        lines.extend([f"- {p}" for p in s["key_points"]])
        lines.append("")
    if s.get("health_advice"):
        lines.append("## Lời khuyên")
        lines.extend([f"- {a}" for a in s["health_advice"]])
        lines.append("")
    if s.get("warnings"):
        lines.append("## ⚠️ Cảnh báo")
        lines.extend([f"- {w}" for w in s["warnings"]])
        lines.append("")
    if s.get("related_conditions"):
        lines.extend(["## Bệnh lý liên quan", ", ".join(f"[[{c}]]" for c in s["related_conditions"]), ""])
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Update Second Brain with new videos")
    parser.add_argument("--check", action="store_true", help="Only check for new videos")
    parser.add_argument("--force", action="store_true", help="Re-process all videos")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold blue]🧠 Second Brain - Update[/]\n"
        f"Channel: {CHANNEL_ID}",
        title="Checking for Updates"
    ))
    
    # Step 1: Check for new videos
    new_videos, all_videos = crawl_new_videos()
    
    if args.check:
        if new_videos:
            console.print(f"\n[yellow]Found {len(new_videos)} new videos:[/]")
            for v in new_videos[:10]:
                console.print(f"  - {v.title[:50]}")
            if len(new_videos) > 10:
                console.print(f"  ... and {len(new_videos) - 10} more")
        else:
            console.print("\n[green]No new videos found.[/]")
        return
    
    if not new_videos and not args.force:
        console.print("\n[green]✓ Already up to date![/]")
        return
    
    # Step 2: Extract transcripts
    if args.force:
        # Re-extract all
        video_ids = [v.id for v in all_videos]
        console.print(f"\n[yellow]Force mode: re-processing all {len(video_ids)} videos[/]")
    else:
        extracted = extract_transcripts(new_videos)
        video_ids = extracted
    
    # Step 3: Generate summaries
    if args.force:
        # Get all transcript IDs
        video_ids = [f.stem for f in Path("data/transcripts").glob("*.json")]
    
    # Find which ones need summaries
    existing_summaries = {f.stem for f in Path("data/summaries").glob("*.json")}
    to_summarize = [vid for vid in video_ids if vid not in existing_summaries or args.force]
    
    if to_summarize:
        process_new_summaries(to_summarize)
    
    # Summary
    total_transcripts = len(list(Path("data/transcripts").glob("*.json")))
    total_summaries = len(list(Path("data/summaries").glob("*.json")))
    
    # Step 4: Regenerate catalog
    console.print("\n[bold cyan]Step 4: Updating catalog...[/]")
    try:
        from generate_catalog import load_summaries, generate_catalog
        summaries = load_summaries()
        catalog = generate_catalog(summaries)
        with open("CATALOG.md", "w", encoding="utf-8") as f:
            f.write(catalog)
        console.print(f"  [green]✓[/] Catalog updated with {len(summaries)} videos")
    except Exception as e:
        console.print(f"  [yellow]Warning: Could not update catalog: {e}[/]")
    
    console.print("\n" + "=" * 50)
    console.print(Panel.fit(
        f"[bold green]✓ Update Complete![/]\n\n"
        f"Total videos: {len(all_videos)}\n"
        f"Transcripts: {total_transcripts}\n"
        f"Summaries: {total_summaries}",
        title="Summary"
    ))


if __name__ == "__main__":
    main()
