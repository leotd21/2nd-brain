"""
Re-process video summaries with enhanced summarization.

This script uses the new multi-chunk, medical-focused summarizer
to generate higher quality summaries.

Usage:
    python reprocess_summaries.py                    # Process all (skip existing)
    python reprocess_summaries.py --force            # Re-process all
    python reprocess_summaries.py --video VIDEO_ID   # Process single video
    python reprocess_summaries.py --sample 5         # Process 5 random videos (for testing)
    python reprocess_summaries.py --low-quality      # Re-process only low quality summaries
    python reprocess_summaries.py --resume           # Resume from last checkpoint
"""

import argparse
import json
import random
import time
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.processor.enhanced_summarizer import EnhancedSummarizer, save_enhanced_summary
from src.processor.categorizer import Categorizer

console = Console()

CHECKPOINT_FILE = Path("data/reprocess_checkpoint.json")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> dict | None:
    """Load existing checkpoint file, or return None."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint: dict):
    """Persist checkpoint to disk after every video."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    checkpoint["updated_at"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


def clear_checkpoint():
    """Remove checkpoint once the run finishes cleanly."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        console.print("[dim]Checkpoint cleared.[/]")


def load_transcript(video_id: str) -> tuple[str, dict]:
    """Load transcript and metadata for a video."""
    transcript_path = Path(f"data/transcripts/{video_id}.json")
    
    if not transcript_path.exists():
        return None, None
    
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Load metadata for title
    metadata_path = Path("data/metadata/dr-tran-van-phuc.json")
    title = video_id
    
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            for v in meta.get("videos", []):
                if v["id"] == video_id:
                    title = v.get("title", video_id)
                    break
    
    return data.get("full_text", ""), {"title": title, "video_id": video_id}


def get_existing_quality(video_id: str) -> float:
    """Get quality score of existing summary."""
    summary_path = Path(f"data/summaries/{video_id}.json")
    
    if not summary_path.exists():
        return 0.0
    
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("quality_score", 0.5)  # Default 0.5 for old summaries
    except:
        return 0.0


def get_all_video_ids() -> list[str]:
    """Get all video IDs with transcripts."""
    transcript_dir = Path("data/transcripts")
    return [f.stem for f in transcript_dir.glob("*.json")]


def process_video(
    video_id: str, 
    summarizer: EnhancedSummarizer,
    categorizer: Categorizer,
    output_dir: Path
) -> dict:
    """Process a single video with enhanced summarization."""
    
    transcript, meta = load_transcript(video_id)
    
    if not transcript:
        return {"status": "error", "message": "No transcript"}
    
    title = meta.get("title", video_id)
    
    # Generate enhanced summary
    summary = summarizer.summarize(transcript, title, video_id)
    
    # Add categories
    text_for_categorization = (
        summary.get("summary", "") + " " + 
        " ".join(summary.get("key_points", []))
    )
    categories = categorizer.categorize(text_for_categorization, title=title)
    summary["categories"] = categories
    
    # Save
    save_enhanced_summary(summary, output_dir)
    
    return {
        "status": "success",
        "quality_score": summary.get("quality_score", 0),
        "coverage": summary.get("transcript_coverage", 0),
        "key_points": len(summary.get("key_points", [])),
        "mechanisms": len(summary.get("mechanisms", []))
    }


def main():
    parser = argparse.ArgumentParser(description="Re-process summaries with enhanced summarizer")
    parser.add_argument("--force", action="store_true", help="Re-process all videos")
    parser.add_argument("--video", type=str, help="Process single video by ID")
    parser.add_argument("--sample", type=int, help="Process N random videos (for testing)")
    parser.add_argument("--low-quality", action="store_true", help="Only re-process low quality (<0.6)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Quality threshold for --low-quality")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold blue]🧠 Enhanced Summary Processor[/]\n\n"
        "Multi-chunk, medical-focused summarization\n"
        "with mechanisms, dosages, and protocols extraction",
        title="Second Brain"
    ))
    
    # Initialize
    summarizer = EnhancedSummarizer()
    categorizer = Categorizer()
    output_dir = Path("data/summaries")
    
    # ---------------------------------------------------------------------------
    # Determine video list — resume or build fresh
    # ---------------------------------------------------------------------------
    checkpoint = load_checkpoint()

    if checkpoint and not args.video and not args.sample:
        if args.resume or _prompt_resume(checkpoint):
            video_ids = checkpoint["video_ids"]
            completed = set(checkpoint.get("completed", []))
            failed = checkpoint.get("failed", {})
            console.print(
                f"\n[yellow]Resuming checkpoint:[/] "
                f"{len(completed)}/{len(video_ids)} done, "
                f"{len(failed)} failed, "
                f"{len(video_ids) - len(completed) - len(failed)} remaining"
            )
        else:
            # User chose to start fresh — discard old checkpoint
            clear_checkpoint()
            checkpoint = None
            completed = set()
            failed = {}
            video_ids = _build_video_list(args, output_dir)
    else:
        completed = set()
        failed = {}
        video_ids = _build_video_list(args, output_dir)

    if not video_ids:
        console.print("[green]✓ Nothing to process![/]")
        return

    # Create / update checkpoint with the full list
    if not checkpoint or checkpoint.get("video_ids") != video_ids:
        checkpoint = {
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "total": len(video_ids),
            "video_ids": video_ids,
            "completed": list(completed),
            "failed": failed,
        }
        save_checkpoint(checkpoint)

    # Remaining = full list minus already completed and failed
    remaining = [v for v in video_ids if v not in completed and v not in failed]
    console.print(f"[cyan]To process: {len(remaining)} videos[/]")

    # ---------------------------------------------------------------------------
    # Process loop
    # ---------------------------------------------------------------------------
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing...", total=len(remaining))

        for video_id in remaining:
            progress.update(task, description=f"[cyan]{video_id}[/]")

            try:
                result = process_video(video_id, summarizer, categorizer, output_dir)
                result["video_id"] = video_id
                results.append(result)

                if result["status"] == "success":
                    completed.add(video_id)
                    console.print(
                        f"  [green]✓[/] {video_id}: "
                        f"quality={result['quality_score']:.0%}, "
                        f"coverage={result['coverage']:.0%}, "
                        f"points={result['key_points']}, "
                        f"mechanisms={result['mechanisms']}"
                    )
                else:
                    failed[video_id] = result.get("message", "Unknown error")
                    console.print(f"  [red]✗[/] {video_id}: {failed[video_id]}")

            except Exception as e:
                failed[video_id] = str(e)
                console.print(f"  [red]✗[/] {video_id}: {str(e)[:50]}")
                results.append({"video_id": video_id, "status": "error", "message": str(e)})

            # Save checkpoint after every video
            checkpoint["completed"] = list(completed)
            checkpoint["failed"] = failed
            save_checkpoint(checkpoint)

            progress.advance(task)
            time.sleep(0.5)  # Rate limiting

    # ---------------------------------------------------------------------------
    # Summary statistics
    # ---------------------------------------------------------------------------
    successful = [r for r in results if r.get("status") == "success"]

    if successful:
        avg_quality = sum(r["quality_score"] for r in successful) / len(successful)
        avg_coverage = sum(r["coverage"] for r in successful) / len(successful)
        avg_points = sum(r["key_points"] for r in successful) / len(successful)

        console.print("\n" + "=" * 50)

        table = Table(title="Processing Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Processed", str(len(video_ids)))
        table.add_row("Successful (this run)", str(len(successful)))
        table.add_row("Failed (this run)", str(len(results) - len(successful)))
        table.add_row("Total completed", str(len(completed)))
        table.add_row("Total failed", str(len(failed)))
        table.add_row("Avg Quality Score", f"{avg_quality:.0%}")
        table.add_row("Avg Coverage", f"{avg_coverage:.0%}")
        table.add_row("Avg Key Points", f"{avg_points:.1f}")

        console.print(table)

    # Clear checkpoint only when all videos are done (no remaining work)
    all_done = len(completed) + len(failed) >= len(video_ids)
    if all_done:
        clear_checkpoint()
    else:
        remaining_count = len(video_ids) - len(completed) - len(failed)
        console.print(
            f"\n[yellow]⚠ Checkpoint kept — {remaining_count} videos still pending. "
            "Run with --resume to continue.[/]"
        )

    console.print("\n[bold green]✓ Processing complete![/]")

    if successful:
        console.print("\n[yellow]💡 Tip: Run 'python build_index.py' to update the search index[/]")


def _prompt_resume(checkpoint: dict) -> bool:
    """Ask user interactively whether to resume the existing checkpoint."""
    completed = len(checkpoint.get("completed", []))
    total = checkpoint.get("total", 0)
    created = checkpoint.get("created_at", "unknown")[:19]
    console.print(
        f"\n[yellow]Found checkpoint from {created}: "
        f"{completed}/{total} completed.[/]"
    )
    answer = console.input("Resume? [Y/n]: ").strip().lower()
    return answer in ("", "y", "yes")


def _build_video_list(args, output_dir: Path) -> list[str]:
    """Build the list of video IDs to process based on CLI args."""
    if args.video:
        console.print(f"\n[cyan]Processing single video: {args.video}[/]")
        return [args.video]

    all_ids = get_all_video_ids()

    if args.sample:
        ids = random.sample(all_ids, min(args.sample, len(all_ids)))
        console.print(f"\n[cyan]Processing {len(ids)} random videos (sample mode)[/]")
        return ids

    if args.low_quality:
        ids = [v for v in all_ids if get_existing_quality(v) < args.threshold]
        console.print(f"\n[cyan]Found {len(ids)} low-quality summaries (< {args.threshold})[/]")
        return ids

    if args.force:
        console.print(f"\n[cyan]Force mode: re-processing all {len(all_ids)} videos[/]")
        return all_ids

    # Default: skip already-processed
    existing = {f.stem for f in output_dir.glob("*.json")}
    ids = [v for v in all_ids if v not in existing]
    console.print(f"\n[cyan]Processing {len(ids)} new videos[/]")
    return ids


if __name__ == "__main__":
    main()
