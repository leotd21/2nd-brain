"""
Health topic categorizer.

Classifies content into health categories using keyword matching and LLM.
"""

import logging
import re
from typing import List, Dict, Set, Tuple

logger = logging.getLogger(__name__)


# Health topic taxonomy
HEALTH_CATEGORIES = {
    "nutrition": {
        "name": "Dinh dưỡng",
        "keywords": [
            "dinh dưỡng", "ăn uống", "thực phẩm", "vitamin", "khoáng chất",
            "protein", "carb", "chất béo", "calo", "giảm cân", "tăng cân",
            "chế độ ăn", "bữa ăn", "thức ăn", "rau", "trái cây", "thịt",
            "cá", "sữa", "đường", "muối", "dầu", "omega", "chất xơ",
            "bổ sung", "supplement", "diet", "food", "eat",
        ],
    },
    "diseases": {
        "name": "Bệnh lý",
        "keywords": [
            "bệnh", "triệu chứng", "chẩn đoán", "điều trị", "thuốc",
            "ung thư", "tiểu đường", "huyết áp", "tim mạch", "gan",
            "thận", "phổi", "dạ dày", "ruột", "xương", "khớp",
            "nhiễm trùng", "viêm", "đau", "sốt", "ho", "cảm",
            "covid", "virus", "vi khuẩn", "disease", "symptom",
        ],
    },
    "lifestyle": {
        "name": "Lối sống",
        "keywords": [
            "tập thể dục", "vận động", "ngủ", "giấc ngủ", "nghỉ ngơi",
            "stress", "căng thẳng", "thư giãn", "thiền", "yoga",
            "thói quen", "lối sống", "sức khỏe", "khỏe mạnh",
            "exercise", "sleep", "lifestyle", "habit", "wellness",
        ],
    },
    "medications": {
        "name": "Thuốc",
        "keywords": [
            "thuốc", "liều", "tác dụng phụ", "kháng sinh", "giảm đau",
            "vitamin", "thực phẩm chức năng", "đơn thuốc", "dược",
            "paracetamol", "aspirin", "ibuprofen", "medicine", "drug",
            "dose", "prescription", "pharmacy",
        ],
    },
    "mental_health": {
        "name": "Sức khỏe tâm thần",
        "keywords": [
            "tâm lý", "tâm thần", "trầm cảm", "lo âu", "stress",
            "căng thẳng", "mất ngủ", "hoảng loạn", "ám ảnh",
            "tự kỷ", "adhd", "depression", "anxiety", "mental",
            "psychology", "therapy", "counseling",
        ],
    },
    "prevention": {
        "name": "Phòng bệnh",
        "keywords": [
            "phòng ngừa", "phòng bệnh", "vaccine", "tiêm chủng",
            "khám sức khỏe", "tầm soát", "xét nghiệm", "kiểm tra",
            "sàng lọc", "prevention", "screening", "checkup",
            "vaccination", "immunization",
        ],
    },
    "emergency": {
        "name": "Cấp cứu",
        "keywords": [
            "cấp cứu", "khẩn cấp", "sơ cứu", "tai nạn", "ngộ độc",
            "đột quỵ", "nhồi máu", "ngừng tim", "chảy máu",
            "bỏng", "gãy xương", "emergency", "first aid", "urgent",
        ],
    },
    "traditional_medicine": {
        "name": "Y học cổ truyền",
        "keywords": [
            "đông y", "thuốc nam", "thảo dược", "châm cứu", "bấm huyệt",
            "massage", "xoa bóp", "cây thuốc", "herbal", "traditional",
            "acupuncture", "alternative medicine",
        ],
    },
    "women_health": {
        "name": "Sức khỏe phụ nữ",
        "keywords": [
            "phụ nữ", "kinh nguyệt", "mang thai", "thai kỳ", "sinh đẻ",
            "mãn kinh", "vú", "tử cung", "buồng trứng", "nội tiết",
            "pregnancy", "menstruation", "menopause", "breast",
        ],
    },
    "children_health": {
        "name": "Sức khỏe trẻ em",
        "keywords": [
            "trẻ em", "trẻ nhỏ", "em bé", "sơ sinh", "tiêm chủng",
            "phát triển", "tăng trưởng", "sữa mẹ", "ăn dặm",
            "children", "baby", "infant", "pediatric", "growth",
        ],
    },
}


class Categorizer:
    """
    Categorizes health content into predefined topics.
    
    Uses keyword matching with optional LLM enhancement.
    
    Example:
        categorizer = Categorizer()
        categories = categorizer.categorize("Vitamin D rất quan trọng cho xương...")
        # Returns: ["nutrition", "prevention"]
    """
    
    def __init__(self, categories: Dict = None, use_llm: bool = False):
        """
        Initialize the categorizer.
        
        Args:
            categories: Custom category definitions (uses default if None)
            use_llm: Whether to use LLM for enhanced categorization
        """
        self.categories = categories or HEALTH_CATEGORIES
        self.use_llm = use_llm
        self._build_keyword_index()
        
    def _build_keyword_index(self):
        """Build inverted index for fast keyword lookup."""
        self.keyword_to_category: Dict[str, Set[str]] = {}
        
        for category, data in self.categories.items():
            for keyword in data["keywords"]:
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_to_category:
                    self.keyword_to_category[keyword_lower] = set()
                self.keyword_to_category[keyword_lower].add(category)
    
    def categorize(
        self, 
        text: str, 
        title: str = "",
        threshold: float = 0.1
    ) -> List[str]:
        """
        Categorize text into health topics.
        
        Args:
            text: Content to categorize
            title: Optional title for additional context
            threshold: Minimum score threshold (0-1)
            
        Returns:
            List of category names, sorted by relevance
        """
        combined_text = f"{title} {text}".lower()
        
        # Count keyword matches per category
        category_scores: Dict[str, float] = {cat: 0 for cat in self.categories}
        
        for keyword, cats in self.keyword_to_category.items():
            # Count occurrences
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', combined_text))
            if count > 0:
                for cat in cats:
                    # Diminishing returns for repeated keywords
                    category_scores[cat] += min(count, 5) / 5
        
        # Normalize scores
        max_score = max(category_scores.values()) if category_scores else 1
        if max_score > 0:
            category_scores = {
                cat: score / max_score 
                for cat, score in category_scores.items()
            }
        
        # Filter by threshold and sort
        matched = [
            (cat, score) 
            for cat, score in category_scores.items() 
            if score >= threshold
        ]
        matched.sort(key=lambda x: x[1], reverse=True)
        
        return [cat for cat, _ in matched]
    
    def categorize_with_scores(
        self, 
        text: str, 
        title: str = ""
    ) -> List[Tuple[str, float]]:
        """
        Categorize text and return scores.
        
        Args:
            text: Content to categorize
            title: Optional title
            
        Returns:
            List of (category, score) tuples, sorted by score
        """
        combined_text = f"{title} {text}".lower()
        
        category_scores: Dict[str, float] = {cat: 0 for cat in self.categories}
        
        for keyword, cats in self.keyword_to_category.items():
            count = len(re.findall(r'\b' + re.escape(keyword) + r'\b', combined_text))
            if count > 0:
                for cat in cats:
                    category_scores[cat] += min(count, 5) / 5
        
        # Normalize
        max_score = max(category_scores.values()) if category_scores else 1
        if max_score > 0:
            category_scores = {
                cat: score / max_score 
                for cat, score in category_scores.items()
            }
        
        results = list(category_scores.items())
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_category_name(self, category_id: str) -> str:
        """Get Vietnamese name for a category."""
        if category_id in self.categories:
            return self.categories[category_id]["name"]
        return category_id
    
    def get_all_categories(self) -> List[Dict]:
        """Get all category definitions."""
        return [
            {
                "id": cat_id,
                "name": data["name"],
                "keyword_count": len(data["keywords"]),
            }
            for cat_id, data in self.categories.items()
        ]


if __name__ == "__main__":
    # Quick test
    categorizer = Categorizer()
    
    test_texts = [
        "Vitamin D rất quan trọng cho xương và hệ miễn dịch",
        "Triệu chứng của bệnh tiểu đường type 2 bao gồm khát nước và đi tiểu nhiều",
        "Cách sơ cứu khi bị bỏng nước sôi",
        "Tập yoga giúp giảm stress và cải thiện giấc ngủ",
    ]
    
    for text in test_texts:
        categories = categorizer.categorize(text)
        print(f"\nText: {text[:50]}...")
        print(f"Categories: {categories}")
