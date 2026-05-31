# 🧠 Second Brain - User Guide

A personal health knowledge base built from Dr. Trần Văn Phúc's YouTube channel.

## Table of Contents

- [Quick Start](#quick-start)
- [Daily Usage](#daily-usage)
- [Updating Content](#updating-content)
- [Scripts Reference](#scripts-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.10+
- 9router running locally at `http://localhost:20128/v1`

### First Time Setup

```bash
# 1. Activate virtual environment
.\venv\Scripts\activate

# 2. Verify everything is ready
python chat.py "Xin chào"
```

---

## Daily Usage

### 💬 Chat with Your Knowledge Base

**Interactive Mode** (recommended):
```bash
python chat.py
```

Commands in chat:
- Type your question in Vietnamese or English
- `sources` - Show detailed sources for last answer
- `quit` or `exit` - Exit chat

**Single Question**:
```bash
python chat.py "Vitamin B12 có tác dụng gì?"
```

**Web Interface** (requires `pip install gradio`):
```bash
python chat.py --web
```
Then open http://localhost:7860 in your browser.

### 📝 Example Questions

```
- Lectin là gì và tại sao nó có hại?
- Làm thế nào để giảm viêm mãn tính?
- Vitamin B1 có vai trò gì trong cơ thể?
- Cortisol ảnh hưởng đến sức khỏe như thế nào?
- Chế độ ăn nào tốt cho người tiểu đường?
- Tại sao cần bổ sung vi chất?
```

---

## Updating Content

### Check for New Videos

```bash
# Just check what's new (no processing)
python update.py --check
```

### Process New Videos

```bash
# Full update: crawl → extract → summarize → index
python update.py
python build_index.py
```

### Force Re-process Everything

```bash
# Re-process all videos (use if summaries need regeneration)
python update.py --force
python build_index.py --rebuild
```

### Recommended Update Schedule

Run weekly to catch new videos:
```bash
python update.py
python build_index.py
```

---

## Scripts Reference

### `chat.py` - Chat Interface

```bash
python chat.py                    # Interactive chat
python chat.py "question"         # Single question
python chat.py --web              # Web interface
python chat.py --no-sources       # Hide source citations
```

### `update.py` - Update Content

```bash
python update.py                  # Process new videos only
python update.py --check          # Check for new videos
python update.py --force          # Re-process all videos
```

### `build_index.py` - Build Search Index

```bash
python build_index.py             # Update index with new content
python build_index.py --rebuild   # Rebuild index from scratch
```

### `process_all.py` - Batch Processing

```bash
python process_all.py             # Process all unprocessed transcripts
```

---

## Configuration

### LLM Settings

Edit the following files to change LLM model:

**`chat.py`** and **`update.py`**:
```python
LOCAL_ENDPOINT = "http://localhost:20128/v1"
LLM_MODEL = "kr/claude-opus-4.5"  # Change model here
```

Available models (check your 9router):
- `kr/claude-opus-4.5` - Best quality
- `kr/claude-sonnet-4.5` - Good balance
- `kr/deepseek-3.2` - Fast and efficient

### Embedding Model

**`build_index.py`** and **`chat.py`**:
```python
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Channel Configuration

**`update.py`**:
```python
CHANNEL_ID = "@BacsiTranVanPhucOfficial"
```

---

## Data Structure

```
2nd-brain/
├── data/
│   ├── metadata/           # Video metadata from YouTube
│   │   └── dr-tran-van-phuc.json
│   ├── transcripts/        # Raw Vietnamese transcripts
│   │   └── {video_id}.json
│   ├── summaries/          # LLM-generated summaries
│   │   ├── {video_id}.json
│   │   └── {video_id}.md   # Obsidian-compatible
│   └── chroma/             # Vector search database
│
├── knowledge_base/         # Obsidian wiki (browse in Obsidian)
│   ├── index.md
│   ├── topics/
│   └── sources/
│
├── chat.py                 # Main chat interface
├── update.py               # Update with new videos
├── build_index.py          # Build search index
└── process_all.py          # Batch processing
```

---

## Troubleshooting

### "Connection refused" error

**Problem**: Can't connect to LLM
**Solution**: Make sure 9router is running at `http://localhost:20128`

### "No collection found" error

**Problem**: Search index not built
**Solution**: Run `python build_index.py`

### Empty or poor quality answers

**Problem**: Not enough relevant content found
**Solutions**:
1. Try rephrasing your question in Vietnamese
2. Use more specific medical terms
3. Rebuild index: `python build_index.py --rebuild`

### Transcript extraction fails

**Problem**: YouTube rate limiting
**Solution**: Wait a few minutes and try again, or run `update.py` later

### Slow first query

**Problem**: Embedding model loading
**Solution**: This is normal for the first query. Subsequent queries will be faster.

---

## Tips for Best Results

1. **Ask in Vietnamese** - The knowledge base is in Vietnamese, so Vietnamese questions work best

2. **Be specific** - "Vitamin B12 thiếu hụt có triệu chứng gì?" works better than "B12"

3. **Check sources** - Type `sources` after an answer to see which videos it came from

4. **Update regularly** - Run `update.py` weekly to get new content

5. **Use Obsidian** - Open `knowledge_base/` folder in Obsidian for a beautiful wiki experience

---

## Statistics

Current knowledge base:
- **110 videos** processed
- **250+ hours** of content
- **1,046 key points** extracted
- **471 health conditions** indexed
- **328 searchable documents**

---

## Support

This is a personal project. For issues:
1. Check the troubleshooting section above
2. Verify 9router is running
3. Try rebuilding the index

---

*Built with ❤️ for better health knowledge*
