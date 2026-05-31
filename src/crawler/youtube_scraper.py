"""
YouTube channel scraper using yt-dlp.

Extracts video metadata from YouTube channels without requiring API keys.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yt_dlp

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for a YouTube video."""
    
    id: str
    title: str
    description: str
    channel_id: str
    channel_name: str
    publish_date: datetime
    duration: int  # seconds
    url: str
    thumbnail_url: str
    view_count: int = 0
    tags: List[str] = field(default_factory=list)
    
    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.id}"
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "channel_id": self.channel_id,
            "channel_name": self.channel_name,
            "publish_date": self.publish_date.isoformat(),
            "duration": self.duration,
            "url": self.url,
            "thumbnail_url": self.thumbnail_url,
            "view_count": self.view_count,
            "tags": self.tags,
        }


class YouTubeScraper:
    """
    Scrapes video metadata from YouTube channels using yt-dlp.
    
    Example:
        scraper = YouTubeScraper()
        videos = scraper.get_channel_videos("@BacsiTranVanPhucOfficial", limit=50)
        for video in videos:
            print(f"{video.title} - {video.publish_date}")
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the scraper.
        
        Args:
            output_dir: Directory to save metadata files (optional)
        """
        self.output_dir = output_dir or Path("./data/metadata")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_channel_videos(
        self, 
        channel_id: str, 
        limit: Optional[int] = None,
        date_after: Optional[datetime] = None
    ) -> List[VideoMetadata]:
        """
        Get all videos from a YouTube channel.
        
        Args:
            channel_id: YouTube channel ID or handle (e.g., "@BacsiTranVanPhucOfficial")
            limit: Maximum number of videos to fetch (None for all)
            date_after: Only fetch videos published after this date
            
        Returns:
            List of VideoMetadata objects
        """
        channel_url = f"https://www.youtube.com/{channel_id}/videos"
        
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "force_generic_extractor": False,
        }
        
        if limit:
            ydl_opts["playlistend"] = limit
            
        videos = []
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Fetching videos from channel: {channel_id}")
                result = ydl.extract_info(channel_url, download=False)
                
                if not result or "entries" not in result:
                    logger.warning(f"No videos found for channel: {channel_id}")
                    return []
                
                for entry in result["entries"]:
                    if entry is None:
                        continue
                        
                    video = self._extract_video_metadata(entry, channel_id)
                    if video:
                        # Filter by date if specified
                        if date_after and video.publish_date < date_after:
                            continue
                        videos.append(video)
                        
                logger.info(f"Found {len(videos)} videos from {channel_id}")
                
        except Exception as e:
            logger.error(f"Error fetching channel videos: {e}")
            raise
            
        return videos
    
    def get_video_details(self, video_id: str) -> Optional[VideoMetadata]:
        """
        Get detailed metadata for a single video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            VideoMetadata object or None if not found
        """
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(video_url, download=False)
                if result:
                    return self._extract_video_metadata(result)
        except Exception as e:
            logger.error(f"Error fetching video details for {video_id}: {e}")
            
        return None
    
    def _extract_video_metadata(
        self, 
        entry: dict, 
        channel_id: str = ""
    ) -> Optional[VideoMetadata]:
        """Extract VideoMetadata from yt-dlp entry."""
        try:
            # Parse upload date
            upload_date_str = entry.get("upload_date", "")
            if upload_date_str:
                publish_date = datetime.strptime(upload_date_str, "%Y%m%d")
            else:
                publish_date = datetime.now()
            
            return VideoMetadata(
                id=entry.get("id", ""),
                title=entry.get("title", "Unknown"),
                description=entry.get("description", ""),
                channel_id=entry.get("channel_id", channel_id),
                channel_name=entry.get("channel", entry.get("uploader", "")),
                publish_date=publish_date,
                duration=int(entry.get("duration", 0) or 0),
                url=entry.get("webpage_url", f"https://www.youtube.com/watch?v={entry.get('id', '')}"),
                thumbnail_url=entry.get("thumbnail", ""),
                view_count=entry.get("view_count", 0) or 0,
                tags=entry.get("tags", []) or [],
            )
        except Exception as e:
            logger.warning(f"Error extracting metadata: {e}")
            return None
    
    def save_metadata(self, videos: List[VideoMetadata], filename: str) -> Path:
        """
        Save video metadata to JSON file.
        
        Args:
            videos: List of VideoMetadata objects
            filename: Output filename (without extension)
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.json"
        
        data = {
            "crawled_at": datetime.now().isoformat(),
            "video_count": len(videos),
            "videos": [v.to_dict() for v in videos],
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved metadata to {output_path}")
        return output_path


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    scraper = YouTubeScraper()
    videos = scraper.get_channel_videos("@BacsiTranVanPhucOfficial", limit=5)
    
    for video in videos:
        print(f"- {video.title} ({video.duration}s)")
