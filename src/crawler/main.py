"""
Main entry point for the crawler module.

Usage:
    python -m src.crawler.main --channel "@BacsiTranVanPhucOfficial" --limit 50
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .youtube_scraper import YouTubeScraper
from .transcript_extractor import TranscriptExtractor

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/crawler.log", encoding="utf-8"),
        ],
    )


def crawl_channel(
    channel_id: str,
    limit: int = 50,
    extract_transcripts: bool = True,
    output_dir: Path = Path("./data"),
) -> dict:
    """
    Crawl a YouTube channel and extract video metadata and transcripts.
    
    Args:
        channel_id: YouTube channel ID or handle
        limit: Maximum number of videos to crawl
        extract_transcripts: Whether to extract transcripts
        output_dir: Output directory for data
        
    Returns:
        Summary of crawl results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata").mkdir(exist_ok=True)
    (output_dir / "transcripts").mkdir(exist_ok=True)
    
    # Initialize components
    scraper = YouTubeScraper(output_dir=output_dir / "metadata")
    extractor = TranscriptExtractor(
        output_dir=output_dir / "transcripts",
        use_whisper_fallback=False,  # Disable Whisper by default
    )
    
    results = {
        "channel_id": channel_id,
        "crawled_at": datetime.now().isoformat(),
        "videos_found": 0,
        "transcripts_extracted": 0,
        "errors": [],
    }
    
    # Fetch video list
    console.print(f"\n[bold blue]Crawling channel:[/] {channel_id}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching video list...", total=None)
        
        try:
            videos = scraper.get_channel_videos(channel_id, limit=limit)
            results["videos_found"] = len(videos)
        except Exception as e:
            console.print(f"[red]Error fetching videos: {e}[/]")
            results["errors"].append(str(e))
            return results
        
        progress.update(task, description=f"Found {len(videos)} videos")
    
    # Display video list
    table = Table(title=f"Videos from {channel_id}")
    table.add_column("ID", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Duration", justify="right")
    table.add_column("Date", justify="right")
    
    for video in videos[:10]:  # Show first 10
        duration_str = f"{video.duration // 60}:{video.duration % 60:02d}"
        date_str = video.publish_date.strftime("%Y-%m-%d")
        table.add_row(video.id, video.title[:50], duration_str, date_str)
    
    if len(videos) > 10:
        table.add_row("...", f"... and {len(videos) - 10} more", "", "")
    
    console.print(table)
    
    # Save metadata
    scraper.save_metadata(videos, f"{channel_id.replace('@', '')}_videos")
    
    # Extract transcripts
    if extract_transcripts and videos:
        console.print("\n[bold blue]Extracting transcripts...[/]")
        
        with Progress(console=console) as progress:
            task = progress.add_task("Processing videos...", total=len(videos))
            
            for video in videos:
                progress.update(task, description=f"Processing: {video.title[:40]}...")
                
                try:
                    transcript = extractor.get_transcript(video.id)
                    if transcript:
                        extractor.save_transcript(transcript)
                        results["transcripts_extracted"] += 1
                except Exception as e:
                    logger.error(f"Error extracting transcript for {video.id}: {e}")
                    results["errors"].append(f"{video.id}: {str(e)}")
                
                progress.advance(task)
    
    # Summary
    console.print("\n[bold green]Crawl Complete![/]")
    console.print(f"  Videos found: {results['videos_found']}")
    console.print(f"  Transcripts extracted: {results['transcripts_extracted']}")
    if results["errors"]:
        console.print(f"  Errors: {len(results['errors'])}")
    
    # Save results summary
    summary_path = output_dir / "crawl_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Crawl YouTube channel for health content"
    )
    parser.add_argument(
        "--channel", "-c",
        default="@BacsiTranVanPhucOfficial",
        help="YouTube channel ID or handle",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=50,
        help="Maximum number of videos to crawl",
    )
    parser.add_argument(
        "--no-transcripts",
        action="store_true",
        help="Skip transcript extraction",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data"),
        help="Output directory",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup
    Path("logs").mkdir(exist_ok=True)
    setup_logging(args.verbose)
    
    # Run crawler
    crawl_channel(
        channel_id=args.channel,
        limit=args.limit,
        extract_transcripts=not args.no_transcripts,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
