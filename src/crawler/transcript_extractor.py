"""
Transcript extractor for YouTube videos.

Extracts transcripts from YouTube captions or uses Whisper for transcription.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A segment of transcript with timing information."""
    
    text: str
    start_time: float  # seconds
    end_time: float    # seconds
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def format_timestamp(self) -> str:
        """Format start time as HH:MM:SS."""
        hours = int(self.start_time // 3600)
        minutes = int((self.start_time % 3600) // 60)
        seconds = int(self.start_time % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


@dataclass
class Transcript:
    """Complete transcript for a video."""
    
    video_id: str
    language: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    is_auto_generated: bool = False
    source: str = "youtube"  # "youtube" or "whisper"
    
    @property
    def full_text(self) -> str:
        """Get the complete transcript as a single string."""
        return " ".join(seg.text for seg in self.segments)
    
    @property
    def duration(self) -> float:
        """Total duration of the transcript."""
        if not self.segments:
            return 0
        return self.segments[-1].end_time
    
    def get_text_at_time(self, time_seconds: float) -> Optional[str]:
        """Get transcript text at a specific timestamp."""
        for segment in self.segments:
            if segment.start_time <= time_seconds <= segment.end_time:
                return segment.text
        return None
    
    def to_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "language": self.language,
            "is_auto_generated": self.is_auto_generated,
            "source": self.source,
            "duration": self.duration,
            "full_text": self.full_text,
            "segments": [
                {
                    "text": seg.text,
                    "start": seg.start_time,
                    "end": seg.end_time,
                }
                for seg in self.segments
            ],
        }


class TranscriptExtractor:
    """
    Extracts transcripts from YouTube videos.
    
    Tries YouTube captions first, falls back to Whisper if needed.
    
    Example:
        extractor = TranscriptExtractor()
        transcript = extractor.get_transcript("dQw4w9WgXcQ")
        print(transcript.full_text)
    """
    
    # Preferred languages in order
    PREFERRED_LANGUAGES = ["vi", "en"]
    
    def __init__(
        self, 
        output_dir: Optional[Path] = None,
        use_whisper_fallback: bool = True
    ):
        """
        Initialize the extractor.
        
        Args:
            output_dir: Directory to save transcript files
            use_whisper_fallback: Whether to use Whisper if no captions available
        """
        self.output_dir = output_dir or Path("./data/transcripts")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_whisper_fallback = use_whisper_fallback
        self._whisper_model = None
        
    def get_transcript(
        self, 
        video_id: str,
        preferred_language: str = "vi"
    ) -> Optional[Transcript]:
        """
        Get transcript for a YouTube video.
        
        Args:
            video_id: YouTube video ID
            preferred_language: Preferred transcript language
            
        Returns:
            Transcript object or None if not available
        """
        # Try YouTube captions first
        transcript = self._get_youtube_transcript(video_id, preferred_language)
        
        if transcript:
            return transcript
            
        # Fall back to Whisper if enabled
        if self.use_whisper_fallback:
            logger.info(f"No captions found, attempting Whisper transcription for {video_id}")
            return self._transcribe_with_whisper(video_id)
            
        return None
    
    def _get_youtube_transcript(
        self, 
        video_id: str,
        preferred_language: str = "vi"
    ) -> Optional[Transcript]:
        """Get transcript from YouTube captions using new API."""
        try:
            # New API: use .list() and .fetch()
            ytt_api = YouTubeTranscriptApi()
            
            # Try to get transcript list
            try:
                transcript_list = ytt_api.list(video_id)
            except Exception as e:
                logger.warning(f"Could not list transcripts for {video_id}: {e}")
                return None
            
            # Try to find preferred language
            languages_to_try = [preferred_language] + [
                lang for lang in self.PREFERRED_LANGUAGES 
                if lang != preferred_language
            ]
            
            # Find best transcript
            selected_transcript = None
            is_auto_generated = False
            language = None
            
            # First try manual transcripts
            for transcript in transcript_list:
                if transcript.language_code in languages_to_try:
                    if not transcript.is_generated:
                        selected_transcript = transcript
                        language = transcript.language_code
                        is_auto_generated = False
                        break
            
            # Fall back to auto-generated
            if selected_transcript is None:
                for transcript in transcript_list:
                    if transcript.language_code in languages_to_try:
                        selected_transcript = transcript
                        language = transcript.language_code
                        is_auto_generated = transcript.is_generated
                        break
            
            # If still nothing, take first available
            if selected_transcript is None and transcript_list:
                selected_transcript = transcript_list[0]
                language = selected_transcript.language_code
                is_auto_generated = selected_transcript.is_generated
            
            if selected_transcript is None:
                logger.warning(f"No transcript found for video {video_id}")
                return None
            
            # Fetch the transcript data
            transcript_data = selected_transcript.fetch()
            
            # Convert to our format
            segments = []
            for item in transcript_data:
                segment = TranscriptSegment(
                    text=self._clean_text(item.text),
                    start_time=item.start,
                    end_time=item.start + item.duration,
                )
                segments.append(segment)
            
            return Transcript(
                video_id=video_id,
                language=language,
                segments=segments,
                is_auto_generated=is_auto_generated,
                source="youtube",
            )
            
        except Exception as e:
            logger.error(f"Error getting transcript for {video_id}: {e}")
            
        return None
    
    def _transcribe_with_whisper(self, video_id: str) -> Optional[Transcript]:
        """
        Transcribe video audio using Whisper.
        
        Note: This requires downloading the audio first.
        """
        try:
            import whisper
            import yt_dlp
            
            # Download audio
            audio_path = self._download_audio(video_id)
            if not audio_path:
                return None
            
            # Load Whisper model (lazy loading)
            if self._whisper_model is None:
                logger.info("Loading Whisper model...")
                self._whisper_model = whisper.load_model("base")
            
            # Transcribe
            logger.info(f"Transcribing {video_id} with Whisper...")
            result = self._whisper_model.transcribe(
                str(audio_path),
                language="vi",
                task="transcribe",
            )
            
            # Convert to our format
            segments = []
            for seg in result.get("segments", []):
                segment = TranscriptSegment(
                    text=self._clean_text(seg["text"]),
                    start_time=seg["start"],
                    end_time=seg["end"],
                )
                segments.append(segment)
            
            # Clean up audio file
            audio_path.unlink(missing_ok=True)
            
            return Transcript(
                video_id=video_id,
                language=result.get("language", "vi"),
                segments=segments,
                is_auto_generated=True,
                source="whisper",
            )
            
        except ImportError:
            logger.error("Whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            logger.error(f"Whisper transcription failed for {video_id}: {e}")
            
        return None
    
    def _download_audio(self, video_id: str) -> Optional[Path]:
        """Download audio from YouTube video."""
        import yt_dlp
        
        output_path = self.output_dir / f"{video_id}.mp3"
        
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(output_path.with_suffix("")),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "quiet": True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            return output_path
        except Exception as e:
            logger.error(f"Failed to download audio for {video_id}: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean transcript text."""
        # Remove multiple spaces
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        # Remove common artifacts
        text = text.replace("[Music]", "").replace("[Applause]", "")
        return text
    
    def save_transcript(self, transcript: Transcript, filename: Optional[str] = None) -> Path:
        """
        Save transcript to file.
        
        Args:
            transcript: Transcript object to save
            filename: Output filename (defaults to video_id)
            
        Returns:
            Path to saved file
        """
        import json
        
        filename = filename or transcript.video_id
        output_path = self.output_dir / f"{filename}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved transcript to {output_path}")
        return output_path


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    extractor = TranscriptExtractor(use_whisper_fallback=False)
    
    # Test with a sample video ID (replace with actual video ID)
    # transcript = extractor.get_transcript("VIDEO_ID_HERE")
    # if transcript:
    #     print(f"Language: {transcript.language}")
    #     print(f"Duration: {transcript.duration}s")
    #     print(f"Text preview: {transcript.full_text[:500]}...")
