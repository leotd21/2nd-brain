"""
Build vector index from summaries for semantic search.

Usage:
    python build_index.py           # Build/update index
    python build_index.py --rebuild # Rebuild from scratch
"""

import argparse
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

console = Console()

# Configuration
CHROMA_PATH = "./data/chroma"
COLLECTION_NAME = "health_knowledge"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_collection(rebuild=False):
    """Get or create ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            console.print("[yellow]Deleted existing collection[/]")
        except:
            pass
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    return collection


def load_summaries():
    """Load all summaries."""
    summaries_dir = Path("data/summaries")
    summaries = []
    
    for f in summaries_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as file:
            summaries.append(json.load(file))
    
    return summaries


def create_documents(summary: dict) -> list:
    """Create searchable documents from a summary."""
    video_id = summary.get("video_id", "")
    title = summary.get("title", "")
    url = summary.get("source_url", "")
    categories = summary.get("categories", [])
    
    documents = []
    
    # Document 1: Full summary
    summary_text = f"""Tiêu đề: {title}

Tóm tắt: {summary.get('summary', '')}

Chủ đề: {', '.join(summary.get('main_topics', []))}

Điểm quan trọng:
{chr(10).join('- ' + p for p in summary.get('key_points', []))}
"""
    
    documents.append({
        "id": f"{video_id}_summary",
        "text": summary_text,
        "metadata": {
            "video_id": video_id,
            "title": title,
            "url": url,
            "type": "summary",
            "categories": ", ".join(categories)
        }
    })
    
    # Document 2: Health advice
    if summary.get("health_advice"):
        advice_text = f"""Tiêu đề: {title}

Lời khuyên sức khỏe:
{chr(10).join('- ' + a for a in summary.get('health_advice', []))}

Cảnh báo:
{chr(10).join('- ' + w for w in summary.get('warnings', []))}
"""
        documents.append({
            "id": f"{video_id}_advice",
            "text": advice_text,
            "metadata": {
                "video_id": video_id,
                "title": title,
                "url": url,
                "type": "advice",
                "categories": ", ".join(categories)
            }
        })
    
    # Document 3: Related conditions
    if summary.get("related_conditions"):
        conditions_text = f"""Tiêu đề: {title}

Bệnh lý liên quan: {', '.join(summary.get('related_conditions', []))}

Tóm tắt: {summary.get('summary', '')[:500]}
"""
        documents.append({
            "id": f"{video_id}_conditions",
            "text": conditions_text,
            "metadata": {
                "video_id": video_id,
                "title": title,
                "url": url,
                "type": "conditions",
                "categories": ", ".join(categories)
            }
        })
    
    return documents


def build_index(rebuild=False):
    """Build the vector index."""
    console.print(Panel.fit(
        "[bold blue]🔍 Building Vector Index[/]\n"
        f"Model: {EMBEDDING_MODEL}",
        title="Second Brain"
    ))
    
    # Load summaries
    summaries = load_summaries()
    console.print(f"[green]✓[/] Loaded {len(summaries)} summaries")
    
    # Get collection
    with console.status("[bold green]Initializing embedding model..."):
        collection = get_collection(rebuild=rebuild)
    
    existing_ids = set(collection.get()["ids"]) if not rebuild else set()
    console.print(f"[green]✓[/] Existing documents: {len(existing_ids)}")
    
    # Create documents
    all_docs = []
    for summary in summaries:
        docs = create_documents(summary)
        for doc in docs:
            if doc["id"] not in existing_ids:
                all_docs.append(doc)
    
    console.print(f"[yellow]→[/] New documents to index: {len(all_docs)}")
    
    if not all_docs:
        console.print("\n[green]Index is up to date![/]")
        return
    
    # Index documents
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Indexing...", total=len(all_docs))
        
        # Batch indexing
        batch_size = 50
        for i in range(0, len(all_docs), batch_size):
            batch = all_docs[i:i+batch_size]
            
            collection.add(
                ids=[d["id"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[d["metadata"] for d in batch]
            )
            
            progress.advance(task, len(batch))
    
    # Summary
    total_docs = collection.count()
    console.print("\n" + "=" * 50)
    console.print(Panel.fit(
        f"[bold green]✓ Index Built![/]\n\n"
        f"Total documents: {total_docs}\n"
        f"Videos indexed: {len(summaries)}",
        title="Complete"
    ))


def main():
    parser = argparse.ArgumentParser(description="Build vector search index")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild index from scratch")
    args = parser.parse_args()
    
    build_index(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
