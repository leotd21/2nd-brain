#!/usr/bin/env python
"""
Full pipeline script to crawl, process, and index health content.

Usage:
    python scripts/run_pipeline.py --channel "@BacsiTranVanPhucOfficial" --limit 50
"""

import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)


def run_pipeline(
    channel_id: str,
    limit: int = 50,
    skip_crawl: bool = False,
    skip_process: bool = False,
    skip_index: bool = False,
):
    """Run the full data pipeline."""
    
    console.print(Panel.fit(
        "[bold blue]🧠 Second Brain Pipeline[/]\n"
        f"Channel: {channel_id}\n"
        f"Limit: {limit} videos",
        title="Starting",
    ))
    
    data_dir = Path("./data")
    
    # Step 1: Crawl
    if not skip_crawl:
        console.print("\n[bold]Step 1: Crawling YouTube channel...[/]")
        from src.crawler.main import crawl_channel
        
        crawl_channel(
            channel_id=channel_id,
            limit=limit,
            extract_transcripts=True,
            output_dir=data_dir,
        )
    else:
        console.print("\n[dim]Step 1: Skipping crawl[/]")
    
    # Step 2: Process
    if not skip_process:
        console.print("\n[bold]Step 2: Processing transcripts...[/]")
        from src.processor.main import process_transcripts
        
        # Find metadata file
        metadata_files = list((data_dir / "metadata").glob("*.json"))
        metadata_file = metadata_files[0] if metadata_files else None
        
        process_transcripts(
            input_dir=data_dir / "transcripts",
            output_dir=data_dir / "summaries",
            metadata_file=metadata_file,
        )
    else:
        console.print("\n[dim]Step 2: Skipping processing[/]")
    
    # Step 3: Index
    if not skip_index:
        console.print("\n[bold]Step 3: Indexing to vector store...[/]")
        from src.storage.vector_store import VectorStore
        import json
        
        store = VectorStore()
        
        # Load summaries and index
        summaries_dir = data_dir / "summaries"
        documents = []
        
        for summary_file in summaries_dir.glob("*.json"):
            with open(summary_file, "r", encoding="utf-8") as f:
                summary = json.load(f)
            
            # Create document for indexing
            text = f"{summary.get('title', '')}\n\n{summary.get('summary', '')}"
            text += "\n\n" + "\n".join(summary.get("key_points", []))
            
            documents.append({
                "id": summary.get("video_id", summary_file.stem),
                "text": text,
                "metadata": {
                    "video_id": summary.get("video_id"),
                    "title": summary.get("title"),
                    "url": summary.get("source_url"),
                    "categories": summary.get("categories", []),
                },
            })
        
        if documents:
            store.add_documents(documents)
            console.print(f"[green]Indexed {len(documents)} documents[/]")
    else:
        console.print("\n[dim]Step 3: Skipping indexing[/]")
    
    console.print(Panel.fit(
        "[bold green]✅ Pipeline complete![/]\n\n"
        "Next steps:\n"
        "• Run `python -m src.app.main` to start chat\n"
        "• Open `knowledge_base/` in Obsidian",
        title="Done",
    ))


def main():
    parser = argparse.ArgumentParser(description="Run Second Brain pipeline")
    parser.add_argument("--channel", "-c", default="@BacsiTranVanPhucOfficial")
    parser.add_argument("--limit", "-l", type=int, default=50)
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-process", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    run_pipeline(
        channel_id=args.channel,
        limit=args.limit,
        skip_crawl=args.skip_crawl,
        skip_process=args.skip_process,
        skip_index=args.skip_index,
    )


if __name__ == "__main__":
    main()
