# 🧠 Second Brain - Personal Health Knowledge Base

A personal healthcare knowledge system built by crawling, summarizing, and organizing content from trusted Vietnamese medical sources, starting with [Dr. Trần Văn Phúc's YouTube channel](https://www.youtube.com/@BacsiTranVanPhucOfficial).

## 📊 Current Stats

| Content | Count |
|---------|-------|
| Videos processed | 110 |
| Total duration | 250+ hours |
| Key points extracted | 1,046 |
| Health conditions indexed | 471 |

## 🚀 Quick Start

```bash
# Activate environment
.\venv\Scripts\activate

# Chat with your knowledge base
python chat.py

# Ask a single question
python chat.py "Vitamin B12 có tác dụng gì?"
```

## 📖 Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Complete usage instructions
- **[Technical Spec](docs/TECHNICAL_SPEC.md)** - Architecture and implementation details

## 🔄 Keep Updated

```bash
# Check for new videos and process them
python update.py

# Rebuild search index
python build_index.py
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  YouTube        │────▶│  Transcript  │────▶│   LLM Summary   │
│  yt-dlp crawler │     │  Extraction  │     │   (9router)     │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Chat/Query    │◀────│   RAG with   │◀────│  ChromaDB +     │
│   Interface     │     │   LLM        │     │  Embeddings     │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## 📁 Project Structure

```
2nd-brain/
├── chat.py               # 💬 Main chat interface
├── update.py             # 🔄 Update with new videos
├── build_index.py        # 🔍 Build search index
├── data/
│   ├── transcripts/      # Raw Vietnamese transcripts
│   ├── summaries/        # LLM-generated summaries
│   └── chroma/           # Vector search database
├── knowledge_base/       # Obsidian-compatible wiki
└── docs/                 # Documentation
```

## 💡 Example Questions

```
- Lectin là gì và tại sao nó có hại?
- Làm thế nào để giảm viêm mãn tính?
- Vitamin B1 có vai trò gì trong cơ thể?
- Cortisol ảnh hưởng đến sức khỏe như thế nào?
```

## ⚠️ Disclaimer

This is a personal knowledge management tool for educational purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult qualified healthcare providers for medical decisions.

## 📄 License

MIT License - Personal use only. Respect content creators' copyrights.
