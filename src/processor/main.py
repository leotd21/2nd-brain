"""
Main entry point for the processor module.

Usage:
    python -m src.processor.main --input ./data/transcripts --output ./data/summaries
"""

import argparse
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from .summarizer import Summarizer
from .categorizer import Categorizer

console = Console()
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def process_transcripts(
    input_dir: Path,
    output_dir: Path,
    metadata_file: Path = None,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> dict:
    """Process all transcripts in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    summarizer = Summarizer(provider=provider, model=model)
    categorizer = Categorizer()
    
    # Load metadata if available
    metadata = {}
    if metadata_file and metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            metadata = {v["id"]: v for v in data.get("videos", [])}
    
    # Find transcript files
    transcript_files = list(input_dir.glob("*.json"))
    console.print(f"Found {len(transcript_files)} transcripts to process")
    
    results = {"processed": 0, "errors": []}
    
    with Progress(console=console) as progress:
        task = progress.add_task("Processing...", total=len(transcript_files))
        
        for transcript_file in transcript_files:
            video_id = transcript_file.stem
            progress.update(task, description=f"Processing {video_id}...")
            
            try:
                # Load transcript
                with open(transcript_file, "r", encoding="utf-8") as f:
                    transcript_data = json.load(f)
                
                # Get metadata
                video_meta = metadata.get(video_id, {})
                title = video_meta.get("title", video_id)
                channel = video_meta.get("channel_name", "Unknown")
                url = video_meta.get("url", f"https://youtube.com/watch?v={video_id}")
                
                # Generate summary
                summary = summarizer.summarize(
                    transcript=transcript_data.get("full_text", ""),
                    title=title,
                    channel=channel,
                    video_id=video_id,
                    source_url=url,
                )
                
                # Add categories
                summary.categories = categorizer.categorize(
                    summary.summary + " ".join(summary.key_points),
                    title=title,
                )
                
                # Save summary
                summarizer.save_summary(summary, output_dir, format="both")
                results["processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing {video_id}: {e}")
                results["errors"].append({"video_id": video_id, "error": str(e)})
            
            progress.advance(task)
    
    console.print(f"\n[green]Processed {results['processed']} videos[/]")
    if results["errors"]:
        console.print(f"[yellow]Errors: {len(results['errors'])}[/]")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Process video transcripts")
    parser.add_argument("--input", "-i", type=Path, default=Path("./data/transcripts"))
    parser.add_argument("--output", "-o", type=Path, default=Path("./data/summaries"))
    parser.add_argument("--metadata", "-m", type=Path, default=None)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    process_transcripts(
        input_dir=args.input,
        output_dir=args.output,
        metadata_file=args.metadata,
        provider=args.provider,
        model=args.model,
    )


if __name__ == "__main__":
    main()
