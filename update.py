"""
Update Second Brain with new videos from the channel.

Usage:
    python update.py              # Check for new videos and process them
    python update.py --force      # Re-process all videos
    python update.py --check      # Just check for new videos, don't process
    python update.py --resume     # Resume interrupted summarization
"""

import argparse
import json
import os
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.crawler.youtube_scraper import YouTubeScraper
from src.crawler.transcript_extractor import TranscriptExtractor
from src.processor.categorizer import Categorizer
from src.processor.enhanced_summarizer import EnhancedSummarizer, save_enhanced_summary

console = Console()

# Configuration
CHANNEL_ID = "@BacsiTranVanPhucOfficial"
CHECKPOINT_FILE = Path("data/update_checkpoint.json")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict | None:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint: dict):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def clear_checkpoint():
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        console.print("[dim]Checkpoint cleared.[/]")


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


def process_new_summaries(video_ids, checkpoint: dict, completed: set, failed: dict):
    """Generate summaries for new videos using enhanced summarizer with checkpoint support."""
    if not video_ids:
        return

    console.print(f"\n[bold cyan]Step 3: Generating {len(video_ids)} summaries...[/]")

    # Load metadata
    with open("data/metadata/dr-tran-van-phuc.json", "r", encoding="utf-8") as f:
        metadata = {v["id"]: v for v in json.load(f).get("videos", [])}

    summarizer = EnhancedSummarizer()
    categorizer = Categorizer()
    summaries_dir = Path("data/summaries")
    summaries_dir.mkdir(exist_ok=True)

    remaining = [v for v in video_ids if v not in completed and v not in failed]
    console.print(f"  [cyan]Remaining: {len(remaining)} (completed: {len(completed)}, failed: {len(failed)})[/]")

    success = 0

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Summarizing...", total=len(remaining))

            for video_id in remaining:
                meta = metadata.get(video_id, {})
                title = meta.get("title", video_id)

                progress.update(task, description=f"[cyan]{title[:30]}...[/]")

                try:
                    with open(f"data/transcripts/{video_id}.json", "r", encoding="utf-8") as f:
                        transcript_data = json.load(f)

                    full_text = transcript_data.get("full_text", "")

                    summary = summarizer.summarize(full_text, title, video_id)
                    summary["source_url"] = meta.get("url", summary.get("source_url", f"https://youtube.com/watch?v={video_id}"))

                    text_for_cat = summary.get("summary", "") + " " + " ".join(summary.get("key_points", []))
                    summary["categories"] = categorizer.categorize(text_for_cat, title=title)

                    save_enhanced_summary(summary, summaries_dir)

                    completed.add(video_id)
                    success += 1

                except Exception as e:
                    failed[video_id] = str(e)
                    console.print(f"  [red]Error {video_id}: {str(e)[:40]}[/]")

                # Save checkpoint after every video
                checkpoint["completed"] = list(completed)
                checkpoint["failed"] = failed
                save_checkpoint(checkpoint)

                progress.advance(task)
                time.sleep(0.5)

    except KeyboardInterrupt:
        checkpoint["completed"] = list(completed)
        checkpoint["failed"] = failed
        save_checkpoint(checkpoint)
        remaining_count = len(video_ids) - len(completed) - len(failed)
        console.print(
            f"\n[yellow]⚠ Interrupted. {len(completed)} completed, "
            f"{remaining_count} remaining.[/]\n"
            "[yellow]Run with --resume to continue.[/]"
        )
        raise  # Re-raise so main() can exit cleanly

    console.print(f"  [green]Summarized: {success} videos[/]")


def main():
    parser = argparse.ArgumentParser(description="Update Second Brain with new videos")
    parser.add_argument("--check", action="store_true", help="Only check for new videos")
    parser.add_argument("--force", action="store_true", help="Re-process all videos")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted summarization")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold blue]🧠 Second Brain - Update[/]\n"
        f"Channel: {CHANNEL_ID}",
        title="Checking for Updates"
    ))

    # ---------------------------------------------------------------------------
    # Resume path: skip crawl/extract, go straight to summarization
    # ---------------------------------------------------------------------------
    checkpoint = load_checkpoint()
    if checkpoint and (args.resume or _prompt_resume(checkpoint)):
        to_summarize = checkpoint["video_ids"]
        completed = set(checkpoint.get("completed", []))
        failed = checkpoint.get("failed", {})
        console.print(
            f"\n[yellow]Resuming checkpoint:[/] "
            f"{len(completed)}/{len(to_summarize)} done, "
            f"{len(to_summarize) - len(completed) - len(failed)} remaining"
        )
        try:
            process_new_summaries(to_summarize, checkpoint, completed, failed)
        except KeyboardInterrupt:
            return
        _finish(checkpoint, to_summarize, completed, failed)
        return

    # Clear stale checkpoint when starting fresh
    if checkpoint:
        clear_checkpoint()

    # ---------------------------------------------------------------------------
    # Normal path
    # ---------------------------------------------------------------------------
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
        video_ids = [v.id for v in all_videos]
        console.print(f"\n[yellow]Force mode: re-processing all {len(video_ids)} videos[/]")
    else:
        extracted = extract_transcripts(new_videos)
        video_ids = extracted
    
    # Step 3: Generate summaries
    if args.force:
        video_ids = [f.stem for f in Path("data/transcripts").glob("*.json")]
    
    existing_summaries = {f.stem for f in Path("data/summaries").glob("*.json")}
    to_summarize = [vid for vid in video_ids if vid not in existing_summaries or args.force]
    
    if to_summarize:
        completed: set = set()
        failed: dict = {}
        checkpoint = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total": len(to_summarize),
            "video_ids": to_summarize,
            "completed": [],
            "failed": {},
        }
        save_checkpoint(checkpoint)
        try:
            process_new_summaries(to_summarize, checkpoint, completed, failed)
        except KeyboardInterrupt:
            return
        _finish(checkpoint, to_summarize, completed, failed)

    # Summary stats
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


def _finish(checkpoint: dict, video_ids: list, completed: set, failed: dict):
    """Clear checkpoint if all done, otherwise remind user to resume."""
    all_done = len(completed) + len(failed) >= len(video_ids)
    if all_done:
        clear_checkpoint()
    else:
        remaining = len(video_ids) - len(completed) - len(failed)
        console.print(
            f"\n[yellow]⚠ Checkpoint kept — {remaining} videos still pending. "
            "Run with --resume to continue.[/]"
        )


def _prompt_resume(checkpoint: dict) -> bool:
    completed = len(checkpoint.get("completed", []))
    total = checkpoint.get("total", 0)
    created = checkpoint.get("created_at", "unknown")[:19]
    console.print(
        f"\n[yellow]Found checkpoint from {created}: "
        f"{completed}/{total} completed.[/]"
    )
    answer = console.input("Resume? [Y/n]: ").strip().lower()
    return answer in ("", "y", "yes")


if __name__ == "__main__":
    main()
