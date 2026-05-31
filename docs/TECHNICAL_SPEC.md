# Technical Specification: Second Brain - Personal Health Knowledge Base

## 1. Overview

### 1.1 Project Goals
- Build an automated pipeline to extract knowledge from Vietnamese healthcare YouTube content
- Create a structured, searchable knowledge base with semantic search capabilities
- Provide a conversational interface for querying health information with source citations

### 1.2 Target Source
- **Primary**: Dr. Trần Văn Phúc's YouTube Channel (@BacsiTranVanPhucOfficial)
- **Future**: Expandable to other trusted medical channels

### 1.3 Key Features
1. Automated YouTube content crawling and transcript extraction
2. LLM-powered summarization and categorization
3. Vector-based semantic search
4. RAG-powered chat interface
5. Obsidian-compatible markdown wiki

---

## 2. System Architecture

### 2.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  yt-dlp     │  │  YouTube    │  │  Whisper (fallback)     │  │
│  │  Downloader │  │  Captions   │  │  Speech-to-Text         │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
│         └────────────────┼─────────────────────┘                │
│                          ▼                                      │
│                 ┌─────────────────┐                             │
│                 │  Raw Transcript │                             │
│                 │  + Metadata     │                             │
│                 └────────┬────────┘                             │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  Text Chunking  │  │  LLM Summary    │  │  Topic          │  │
│  │  & Cleaning     │──▶  Generation     │──▶  Classification │  │
│  └─────────────────┘  └─────────────────┘  └────────┬────────┘  │
│                                                      │          │
│  ┌─────────────────┐  ┌─────────────────┐           │          │
│  │  Embedding      │◀─│  Structured     │◀──────────┘          │
│  │  Generation     │  │  Note Creation  │                      │
│  └────────┬────────┘  └────────┬────────┘                      │
└───────────┼────────────────────┼────────────────────────────────┘
            ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                       STORAGE LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  ChromaDB       │  │  Markdown       │  │  SQLite         │  │
│  │  Vector Store   │  │  Wiki Files     │  │  Metadata DB    │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
└───────────┼────────────────────┼────────────────────┼───────────┘
            └────────────────────┼────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  RAG Engine     │  │  Chat Interface │  │  Web UI         │  │
│  │  (LangChain)    │──▶  (CLI/Gradio)   │  │  (Optional)     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

1. **Crawl**: Extract video list from YouTube channel
2. **Download**: Get transcripts (captions or Whisper transcription)
3. **Process**: Clean, chunk, and summarize with LLM
4. **Categorize**: Classify into health topics
5. **Store**: Save to vector DB + markdown files
6. **Query**: RAG retrieval + LLM response generation

---

## 3. Component Specifications

### 3.1 Crawler Module (`src/crawler/`)

#### 3.1.1 YouTube Scraper
```python
# Responsibilities:
# - Fetch channel video list via YouTube Data API or yt-dlp
# - Extract video metadata (title, description, duration, publish date)
# - Handle pagination for channels with many videos
# - Track already-processed videos to avoid duplicates

# Key Classes:
class YouTubeScraper:
    def get_channel_videos(channel_id: str) -> List[VideoMetadata]
    def get_video_details(video_id: str) -> VideoMetadata
    def check_new_videos(channel_id: str, last_check: datetime) -> List[str]
```

#### 3.1.2 Transcript Extractor
```python
# Responsibilities:
# - Download Vietnamese captions if available
# - Fall back to Whisper transcription if no captions
# - Handle timestamp alignment
# - Clean and normalize Vietnamese text

# Key Classes:
class TranscriptExtractor:
    def get_transcript(video_id: str) -> Transcript
    def transcribe_audio(audio_path: str) -> Transcript  # Whisper fallback
```

### 3.2 Processor Module (`src/processor/`)

#### 3.2.1 Text Processor
```python
# Responsibilities:
# - Clean transcript text (remove filler words, normalize)
# - Chunk long transcripts into manageable segments
# - Preserve context across chunks

# Configuration:
CHUNK_SIZE = 2000  # characters
CHUNK_OVERLAP = 200  # characters
```

#### 3.2.2 Summarizer
```python
# Responsibilities:
# - Generate structured summaries using LLM
# - Extract key health topics, advice, and warnings
# - Create markdown-formatted notes

# Output Structure:
class VideoSummary:
    title: str
    source_url: str
    main_topics: List[str]
    summary: str  # 2-3 paragraphs
    key_points: List[str]  # Bullet points
    health_advice: List[str]
    warnings: List[str]  # Important cautions
    related_conditions: List[str]
    timestamps: Dict[str, str]  # Topic -> timestamp
```

#### 3.2.3 Categorizer
```python
# Health Topic Taxonomy:
HEALTH_CATEGORIES = {
    "nutrition": ["diet", "vitamins", "supplements", "food"],
    "diseases": ["symptoms", "diagnosis", "treatment", "prevention"],
    "lifestyle": ["exercise", "sleep", "stress", "habits"],
    "medications": ["drugs", "side-effects", "interactions"],
    "mental_health": ["anxiety", "depression", "psychology"],
    "prevention": ["screening", "vaccines", "checkups"],
    "emergency": ["first-aid", "urgent-care", "warning-signs"],
    "traditional_medicine": ["herbal", "acupuncture", "alternative"],
}
```

### 3.3 Storage Module (`src/storage/`)

#### 3.3.1 Vector Store
```python
# Technology: ChromaDB (local, no external dependencies)
# Embedding Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#                  (supports Vietnamese)

# Collections:
# - video_summaries: Full video summary embeddings
# - video_chunks: Individual chunk embeddings for detailed retrieval
# - topics: Topic-level embeddings for browsing
```

#### 3.3.2 Markdown Wiki
```
knowledge_base/
├── index.md                    # Main index with links
├── topics/
│   ├── nutrition/
│   │   ├── _index.md          # Topic overview
│   │   ├── vitamin-d.md
│   │   └── healthy-diet.md
│   ├── diseases/
│   │   ├── _index.md
│   │   ├── diabetes.md
│   │   └── hypertension.md
│   └── ...
├── sources/
│   └── dr-tran-van-phuc/
│       ├── _index.md          # Source overview
│       └── videos/
│           ├── video-001.md   # Individual video notes
│           └── video-002.md
└── tags/
    └── _tags.md               # Tag index
```

#### 3.3.3 Metadata Database (SQLite)
```sql
-- Videos table
CREATE TABLE videos (
    id TEXT PRIMARY KEY,
    channel_id TEXT,
    title TEXT,
    description TEXT,
    publish_date DATETIME,
    duration INTEGER,
    url TEXT,
    transcript_path TEXT,
    summary_path TEXT,
    processed_at DATETIME,
    status TEXT  -- 'pending', 'processed', 'failed'
);

-- Topics table
CREATE TABLE topics (
    id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    description TEXT
);

-- Video-Topic mapping
CREATE TABLE video_topics (
    video_id TEXT,
    topic_id INTEGER,
    relevance_score FLOAT,
    FOREIGN KEY (video_id) REFERENCES videos(id),
    FOREIGN KEY (topic_id) REFERENCES topics(id)
);
```

### 3.4 Application Module (`src/app/`)

#### 3.4.1 RAG Engine
```python
# Framework: LangChain
# Retrieval Strategy: Hybrid (semantic + keyword)

class RAGEngine:
    def query(question: str, top_k: int = 5) -> RAGResponse
    def get_sources(question: str) -> List[Source]
    
class RAGResponse:
    answer: str
    sources: List[Source]
    confidence: float
    related_topics: List[str]
```

#### 3.4.2 Chat Interface
```python
# Options:
# 1. CLI interface (default)
# 2. Gradio web UI (optional)
# 3. Streamlit dashboard (optional)

# Features:
# - Conversational health Q&A
# - Source citations with video links
# - Topic browsing
# - Search history
```

---

## 4. Technology Stack

### 4.1 Core Dependencies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Language | Python | 3.10+ | Main language |
| Crawler | yt-dlp | latest | YouTube download |
| Transcription | openai-whisper | latest | Speech-to-text fallback |
| LLM | OpenAI API / Anthropic | - | Summarization & chat |
| Embeddings | sentence-transformers | latest | Vector embeddings |
| Vector DB | ChromaDB | latest | Semantic search |
| RAG Framework | LangChain | latest | RAG orchestration |
| Database | SQLite | built-in | Metadata storage |
| UI | Gradio | latest | Web interface |

### 4.2 Vietnamese Language Support

- **Embedding Model**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Text Processing**: `underthesea` (Vietnamese NLP library)
- **LLM**: GPT-4 / Claude (both handle Vietnamese well)

---

## 5. Configuration

### 5.1 Settings File (`config/settings.yaml`)

```yaml
# API Keys (use environment variables in production)
openai_api_key: ${OPENAI_API_KEY}
anthropic_api_key: ${ANTHROPIC_API_KEY}
youtube_api_key: ${YOUTUBE_API_KEY}

# Crawler settings
crawler:
  channels:
    - id: "@BacsiTranVanPhucOfficial"
      name: "Dr. Trần Văn Phúc"
  max_videos_per_run: 50
  check_interval_hours: 24

# Processing settings
processor:
  llm_provider: "openai"  # or "anthropic"
  llm_model: "gpt-4o-mini"
  chunk_size: 2000
  chunk_overlap: 200
  
# Storage settings
storage:
  vector_db_path: "./data/chroma"
  wiki_path: "./knowledge_base"
  sqlite_path: "./data/metadata.db"

# Embedding settings
embeddings:
  model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  
# RAG settings
rag:
  top_k: 5
  similarity_threshold: 0.7
```

---

## 6. API Specifications

### 6.1 Internal APIs

#### Crawler API
```python
# GET channel videos
crawler.get_videos(channel_id: str, limit: int = 100) -> List[Video]

# GET transcript
crawler.get_transcript(video_id: str) -> Transcript
```

#### Processor API
```python
# Process single video
processor.process_video(video_id: str) -> ProcessedVideo

# Batch process
processor.process_batch(video_ids: List[str]) -> BatchResult
```

#### Query API
```python
# Ask question
rag.query(question: str) -> Answer

# Browse topics
rag.get_topics() -> List[Topic]

# Get video summary
rag.get_video_summary(video_id: str) -> Summary
```

---

## 7. Data Models

### 7.1 Video Metadata
```python
@dataclass
class VideoMetadata:
    id: str
    channel_id: str
    title: str
    description: str
    publish_date: datetime
    duration: int  # seconds
    url: str
    thumbnail_url: str
    view_count: int
    tags: List[str]
```

### 7.2 Transcript
```python
@dataclass
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float

@dataclass
class Transcript:
    video_id: str
    language: str
    segments: List[TranscriptSegment]
    full_text: str
```

### 7.3 Processed Content
```python
@dataclass
class ProcessedVideo:
    metadata: VideoMetadata
    transcript: Transcript
    summary: VideoSummary
    chunks: List[TextChunk]
    embeddings: List[np.ndarray]
    topics: List[str]
    markdown_path: str
```

---

## 8. Error Handling

### 8.1 Retry Strategy
- Network errors: Exponential backoff (3 retries)
- Rate limits: Respect API limits, queue requests
- Transcription failures: Log and skip, mark for manual review

### 8.2 Logging
```python
# Log levels:
# - INFO: Normal operations
# - WARNING: Recoverable issues
# - ERROR: Failed operations
# - DEBUG: Detailed debugging

# Log format:
"%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 9. Security Considerations

1. **API Keys**: Store in environment variables, never commit
2. **Data Privacy**: Local storage only, no external transmission
3. **Content Rights**: Summarize/paraphrase, don't copy verbatim
4. **Medical Disclaimer**: Always display disclaimer in UI

---

## 10. Future Enhancements

1. **Multi-source Support**: Add more trusted medical channels
2. **Fact Verification**: Cross-reference with medical databases
3. **Personalization**: Track user health interests
4. **Mobile App**: React Native companion app
5. **Voice Interface**: Voice-based health queries
6. **Export**: Export to Anki flashcards for learning

---

## 11. Development Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Project setup and dependencies
- [ ] Basic YouTube crawler
- [ ] Transcript extraction

### Phase 2: Processing (Week 3-4)
- [ ] LLM summarization pipeline
- [ ] Topic categorization
- [ ] Markdown generation

### Phase 3: Storage (Week 5)
- [ ] ChromaDB integration
- [ ] SQLite metadata store
- [ ] Wiki structure

### Phase 4: Application (Week 6-7)
- [ ] RAG engine
- [ ] CLI interface
- [ ] Basic web UI

### Phase 5: Polish (Week 8)
- [ ] Testing and bug fixes
- [ ] Documentation
- [ ] Performance optimization
