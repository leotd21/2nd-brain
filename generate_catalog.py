"""
Generate a catalog of all video summaries in markdown format.

Usage:
    python generate_catalog.py
"""

import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_summaries():
    """Load all summaries and metadata."""
    summaries_dir = Path("data/summaries")
    metadata_file = Path("data/metadata/dr-tran-van-phuc.json")
    
    # Load metadata for publish dates
    publish_dates = {}
    if metadata_file.exists():
        with open(metadata_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for v in data.get("videos", []):
                publish_dates[v["id"]] = v.get("publish_date", "")
    
    summaries = []
    for f in summaries_dir.glob("*.json"):
        with open(f, "r", encoding="utf-8") as file:
            s = json.load(file)
            s["publish_date"] = publish_dates.get(s.get("video_id", ""), "")
            summaries.append(s)
    
    # Sort by publish date (newest first)
    summaries.sort(key=lambda x: x.get("publish_date", ""), reverse=True)
    
    return summaries


def generate_catalog(summaries):
    """Generate markdown catalog."""
    lines = [
        "# 📚 Video Catalog",
        "",
        f"Total: **{len(summaries)} videos** from Dr. Trần Văn Phúc's channel",
        "",
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## 📋 All Videos",
        "",
        "| # | Title | Categories | Summary |",
        "|---|-------|------------|---------|",
    ]
    
    for i, s in enumerate(summaries, 1):
        video_id = s.get("video_id", "")
        title = s.get("title", "Untitled")
        categories = s.get("categories", [])
        
        # Create link to summary file
        summary_link = f"[📄](data/summaries/{video_id}.md)"
        
        # Format categories as badges
        cat_str = " ".join([f"`{c}`" for c in categories[:3]])
        
        # Escape pipe characters in title
        title_escaped = title.replace("|", "\\|")
        
        lines.append(f"| {i} | {title_escaped} | {cat_str} | {summary_link} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## 📊 By Category",
        "",
    ])
    
    # Group by category
    by_category = defaultdict(list)
    for s in summaries:
        for cat in s.get("categories", []):
            by_category[cat].append(s)
    
    # Category names in Vietnamese
    cat_names = {
        "nutrition": "🥗 Dinh dưỡng",
        "diseases": "🏥 Bệnh lý",
        "lifestyle": "🏃 Lối sống",
        "medications": "💊 Thuốc",
        "mental_health": "🧠 Sức khỏe tâm thần",
        "prevention": "🛡️ Phòng bệnh",
        "emergency": "🚨 Cấp cứu",
        "children_health": "👶 Sức khỏe trẻ em",
        "women_health": "👩 Sức khỏe phụ nữ",
        "traditional_medicine": "🌿 Y học cổ truyền",
    }
    
    # Sort categories by count
    sorted_cats = sorted(by_category.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cat, videos in sorted_cats:
        cat_display = cat_names.get(cat, cat)
        lines.append(f"### {cat_display} ({len(videos)} videos)")
        lines.append("")
        
        for s in videos[:10]:  # Show top 10 per category
            video_id = s.get("video_id", "")
            title = s.get("title", "Untitled")
            lines.append(f"- [{title}](data/summaries/{video_id}.md)")
        
        if len(videos) > 10:
            lines.append(f"- *... and {len(videos) - 10} more*")
        
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## 🔍 Quick Links",
        "",
        "- [User Guide](docs/USER_GUIDE.md)",
        "- [Technical Spec](docs/TECHNICAL_SPEC.md)",
        "- [Knowledge Base Wiki](knowledge_base/index.md)",
        "",
        "---",
        "",
        "*Generated automatically. Run `python generate_catalog.py` to update.*",
    ])
    
    return "\n".join(lines)


def main():
    print("Generating catalog...")
    
    summaries = load_summaries()
    print(f"Loaded {len(summaries)} summaries")
    
    catalog = generate_catalog(summaries)
    
    # Save catalog
    output_path = Path("CATALOG.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(catalog)
    
    print(f"Catalog saved to {output_path}")


if __name__ == "__main__":
    main()
