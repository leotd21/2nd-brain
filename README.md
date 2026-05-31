# 🧠 Second Brain - Personal Health Knowledge Base

A personal healthcare knowledge system built by crawling, summarizing, and organizing content from trusted Vietnamese medical sources, starting with [Dr. Trần Văn Phúc's YouTube channel](https://www.youtube.com/@BacsiTranVanPhucOfficial).

## 🎯 Purpose

Build a searchable, AI-powered personal health knowledge base that:
- Aggregates health information from trusted Vietnamese medical professionals
- Provides quick answers to health questions with source citations
- Organizes medical knowledge by topics for easy browsing

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  YouTube API /  │────▶│  Transcript  │────▶│   LLM Summary   │
│  yt-dlp crawler │     │  Extraction  │     │   & Structuring │
└─────────────────┘     └──────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Chat/Query    │◀────│   RAG with   │◀────│  Vector DB +    │
│   Interface     │     │   LLM        │     │  Markdown Wiki  │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

## 📁 Project Structure

```
2nd-brain/
├── src/
│   ├── crawler/          # YouTube content extraction
│   ├── processor/        # LLM summarization & categorization
│   ├── storage/          # Vector DB & file management
│   └── app/              # RAG engine & chat interface
├── knowledge_base/       # Markdown wiki (Obsidian-compatible)
├── data/                 # Raw data & embeddings
├── config/               # Configuration files
├── tests/                # Test files
└── docs/                 # Documentation & specs
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys
cp config/settings.example.yaml config/settings.yaml
# Edit settings.yaml with your API keys

# 3. Crawl channel content
python -m src.crawler.main --channel "@BacsiTranVanPhucOfficial"

# 4. Process and summarize
python -m src.processor.main

# 5. Start chat interface
python -m src.app.main
```

## ⚠️ Disclaimer

This is a personal knowledge management tool for educational purposes only. 
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult qualified healthcare providers for medical decisions.

## 📄 License

MIT License - Personal use only. Respect content creators' copyrights.
