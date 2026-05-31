"""Tests for crawler module."""

import pytest
from pathlib import Path
from datetime import datetime

from src.crawler.youtube_scraper import YouTubeScraper, VideoMetadata


class TestVideoMetadata:
    """Tests for VideoMetadata dataclass."""
    
    def test_youtube_url(self):
        """Test YouTube URL generation."""
        video = VideoMetadata(
            id="abc123",
            title="Test Video",
            description="Test",
            channel_id="channel1",
            channel_name="Test Channel",
            publish_date=datetime.now(),
            duration=300,
            url="https://youtube.com/watch?v=abc123",
            thumbnail_url="",
        )
        assert video.youtube_url == "https://www.youtube.com/watch?v=abc123"
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        video = VideoMetadata(
            id="abc123",
            title="Test Video",
            description="Test description",
            channel_id="channel1",
            channel_name="Test Channel",
            publish_date=datetime(2024, 1, 15),
            duration=300,
            url="https://youtube.com/watch?v=abc123",
            thumbnail_url="https://img.youtube.com/abc123.jpg",
            view_count=1000,
            tags=["health", "nutrition"],
        )
        
        data = video.to_dict()
        
        assert data["id"] == "abc123"
        assert data["title"] == "Test Video"
        assert data["duration"] == 300
        assert data["tags"] == ["health", "nutrition"]


class TestYouTubeScraper:
    """Tests for YouTubeScraper class."""
    
    def test_init(self, tmp_path):
        """Test scraper initialization."""
        scraper = YouTubeScraper(output_dir=tmp_path)
        assert scraper.output_dir == tmp_path
    
    def test_save_metadata(self, tmp_path):
        """Test saving metadata to file."""
        scraper = YouTubeScraper(output_dir=tmp_path)
        
        videos = [
            VideoMetadata(
                id="vid1",
                title="Video 1",
                description="",
                channel_id="ch1",
                channel_name="Channel",
                publish_date=datetime.now(),
                duration=100,
                url="",
                thumbnail_url="",
            ),
        ]
        
        output_path = scraper.save_metadata(videos, "test_videos")
        
        assert output_path.exists()
        assert output_path.name == "test_videos.json"


# Integration tests (require network, skip by default)
@pytest.mark.skip(reason="Requires network access")
class TestYouTubeScraperIntegration:
    """Integration tests for YouTubeScraper."""
    
    def test_get_channel_videos(self):
        """Test fetching videos from a real channel."""
        scraper = YouTubeScraper()
        videos = scraper.get_channel_videos("@BacsiTranVanPhucOfficial", limit=3)
        
        assert len(videos) > 0
        assert all(isinstance(v, VideoMetadata) for v in videos)
