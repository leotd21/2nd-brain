"""Tests for categorizer module."""

import pytest

from src.processor.categorizer import Categorizer, HEALTH_CATEGORIES


class TestCategorizer:
    """Tests for Categorizer class."""
    
    @pytest.fixture
    def categorizer(self):
        return Categorizer()
    
    def test_categorize_nutrition(self, categorizer):
        """Test categorization of nutrition content."""
        text = "Vitamin D rất quan trọng cho xương và hệ miễn dịch"
        categories = categorizer.categorize(text)
        
        assert "nutrition" in categories
    
    def test_categorize_diseases(self, categorizer):
        """Test categorization of disease content."""
        text = "Triệu chứng của bệnh tiểu đường type 2 bao gồm khát nước"
        categories = categorizer.categorize(text)
        
        assert "diseases" in categories
    
    def test_categorize_emergency(self, categorizer):
        """Test categorization of emergency content."""
        text = "Cách sơ cứu khi bị bỏng nước sôi"
        categories = categorizer.categorize(text)
        
        assert "emergency" in categories
    
    def test_categorize_lifestyle(self, categorizer):
        """Test categorization of lifestyle content."""
        text = "Tập yoga giúp giảm stress và cải thiện giấc ngủ"
        categories = categorizer.categorize(text)
        
        assert "lifestyle" in categories
    
    def test_categorize_with_scores(self, categorizer):
        """Test categorization with scores."""
        text = "Vitamin D và calcium cho xương chắc khỏe"
        results = categorizer.categorize_with_scores(text)
        
        assert len(results) > 0
        assert all(isinstance(score, float) for _, score in results)
    
    def test_get_category_name(self, categorizer):
        """Test getting Vietnamese category name."""
        assert categorizer.get_category_name("nutrition") == "Dinh dưỡng"
        assert categorizer.get_category_name("diseases") == "Bệnh lý"
    
    def test_get_all_categories(self, categorizer):
        """Test getting all categories."""
        categories = categorizer.get_all_categories()
        
        assert len(categories) == len(HEALTH_CATEGORIES)
        assert all("id" in cat and "name" in cat for cat in categories)
