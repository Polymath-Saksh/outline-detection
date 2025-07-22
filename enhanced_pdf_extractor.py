
# ENHANCED PDF OUTLINE EXTRACTION - UNSUPERVISED IMPROVEMENTS
# ===========================================================
from utils import *
import pymupdf #type: ignore[import]
import pandas as pd #type: ignore[import]
import numpy as np
import re
from collections import Counter, defaultdict
import string
from typing import Dict, List, Tuple, Optional

# 1. CROSS-PAGE FONT ANALYSIS - Major improvement for consistency
def analyze_fonts_globally(df: pd.DataFrame) -> Dict:
    """Analyze font patterns across entire document for better consistency"""

    # Font size clustering across all pages
    all_sizes = df['size'].values
    size_clusters = {}

    # Use simple clustering based on size gaps
    unique_sizes = sorted(df['size'].unique(), reverse=True)
    clusters = []
    current_cluster = [unique_sizes[0]]

    for i in range(1, len(unique_sizes)):
        # If size difference is less than 10% of current size, group together
        if (current_cluster[0] - unique_sizes[i]) / current_cluster[0] < 0.1:
            current_cluster.append(unique_sizes[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [unique_sizes[i]]

    clusters.append(current_cluster)

    # Assign cluster rankings
    for rank, cluster in enumerate(clusters, 1):
        for size in cluster:
            size_clusters[size] = rank

    # Font family analysis
    font_patterns = Counter(df['font'].values)
    common_fonts = {font: count for font, count in font_patterns.items()}

    return {
        'size_clusters': size_clusters,
        'font_frequency': common_fonts,
        'total_spans': len(df)
    }

# 2. ENHANCED POSITION FEATURES - Using bounding box data
def extract_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract position-based features from bounding boxes"""

    df_enhanced = df.copy()

    # Extract position metrics from bbox [x0, y0, x1, y1]
    df_enhanced['x0'] = df_enhanced['bbox'].apply(lambda x: x[0])
    df_enhanced['y0'] = df_enhanced['bbox'].apply(lambda x: x[1])
    df_enhanced['x1'] = df_enhanced['bbox'].apply(lambda x: x[2])
    df_enhanced['y1'] = df_enhanced['bbox'].apply(lambda x: x[3])

    # Position features
    df_enhanced['width'] = df_enhanced['x1'] - df_enhanced['x0']
    df_enhanced['height'] = df_enhanced['y1'] - df_enhanced['y0']
    df_enhanced['left_margin'] = df_enhanced['x0']
    df_enhanced['center_x'] = (df_enhanced['x0'] + df_enhanced['x1']) / 2

    # Page-relative positioning
    for page in df_enhanced['page'].unique():
        page_mask = df_enhanced['page'] == page
        page_data = df_enhanced[page_mask]

        # Calculate relative positions within page
        df_enhanced.loc[page_mask, 'left_margin_norm'] =             (page_data['left_margin'] - page_data['left_margin'].min()) /             (page_data['left_margin'].max() - page_data['left_margin'].min() + 1e-6)

        df_enhanced.loc[page_mask, 'y_position_norm'] =             page_data['y0'] / page_data['y0'].max()

    return df_enhanced

# 3. LINGUISTIC PATTERN DETECTION - Rule-based approach
def analyze_linguistic_patterns(text: str) -> Dict:
    """Analyze text for heading-like linguistic patterns"""

    patterns = {
        'has_numbers': bool(re.search(r'\d', text)),
        'starts_with_number': bool(re.match(r'^\d', text.strip())),
        'has_section_pattern': bool(re.match(r'^\d+\.\d*', text.strip())),
        'all_caps': text.isupper() and len(text) > 1,
        'title_case': text.istitle(),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]),
        'has_colon': ':' in text,
        'ends_with_period': text.strip().endswith('.'),
        'starts_with_article': text.lower().strip().startswith(('the ', 'a ', 'an ')),
    }

    # Common heading keywords
    heading_keywords = {
        'introduction', 'conclusion', 'abstract', 'summary', 'background',
        'methodology', 'method', 'results', 'discussion', 'chapter',
        'section', 'overview', 'analysis', 'findings', 'references'
    }

    words_lower = [word.lower().strip(string.punctuation) for word in text.split()]
    patterns['has_heading_keyword'] = any(word in heading_keywords for word in words_lower)

    # Calculate pattern score
    score = 0
    if patterns['has_section_pattern']: score += 3
    if patterns['has_numbers']: score += 1
    if patterns['title_case'] or patterns['all_caps']: score += 2
    if patterns['has_heading_keyword']: score += 2
    if patterns['word_count'] <= 10: score += 1  # Headings are usually short
    if not patterns['ends_with_period']: score += 1  # Headings rarely end with periods
    if not patterns['starts_with_article']: score += 1  # Headings rarely start with articles

    patterns['linguistic_score'] = score
    return patterns

# 4. CONTEXT-AWARE FILTERING - Analyze surrounding text
def analyze_context(df: pd.DataFrame, idx: int, window: int = 2) -> Dict:
    """Analyze text context around a potential heading"""

    current_row = df.iloc[idx]
    page = current_row['page']

    # Get surrounding spans on the same page
    page_spans = df[df['page'] == page].sort_values('y0')
    current_idx_in_page = page_spans.index.get_loc(idx)

    context = {
        'before_spans': [],
        'after_spans': [],
        'is_page_start': current_idx_in_page < window,
        'is_page_end': current_idx_in_page >= len(page_spans) - window,
    }

    # Get before and after text
    start_idx = max(0, current_idx_in_page - window)
    end_idx = min(len(page_spans), current_idx_in_page + window + 1)

    for i in range(start_idx, current_idx_in_page):
        context['before_spans'].append(page_spans.iloc[i]['text'])

    for i in range(current_idx_in_page + 1, end_idx):
        context['after_spans'].append(page_spans.iloc[i]['text'])

    # Analyze vertical spacing (indicates paragraph breaks)
    if current_idx_in_page > 0:
        prev_span = page_spans.iloc[current_idx_in_page - 1]
        vertical_gap = current_row['y0'] - prev_span['y1']
        context['large_gap_before'] = vertical_gap > current_row['height'] * 1.5
    else:
        context['large_gap_before'] = True

    return context

# 5. ENHANCED CLASSIFICATION WITH MULTIPLE FEATURES
def classify_heading_enhanced(row: pd.Series, font_analysis: Dict, 
                            linguistic_patterns: Dict, context: Dict) -> Tuple[str, float]:
    """Enhanced heading classification using multiple feature types"""

    scores = {
        'title': 0,
        'h1': 0,
        'h2': 0,
        'h3': 0,
        'none': 0
    }

    # Font size scoring
    size_cluster = font_analysis['size_clusters'].get(row['size'], 999)
    if size_cluster == 1:  # Largest font cluster
        if row['page'] == 1:
            scores['title'] += 5
        else:
            scores['h1'] += 4
    elif size_cluster == 2:
        scores['h1'] += 3
        scores['h2'] += 2
    elif size_cluster == 3:
        scores['h2'] += 3
        scores['h3'] += 2
    else:
        scores['none'] += 2

    # Style scoring
    if row['is_bold']:
        scores['title'] += 2
        scores['h1'] += 2
        scores['h2'] += 1
        scores['h3'] += 1

    # Position scoring
    if row.get('left_margin_norm', 0) < 0.1:  # Left-aligned
        scores['title'] += 1
        scores['h1'] += 1

    if context.get('large_gap_before', False):  # Has spacing before
        scores['title'] += 2
        scores['h1'] += 2
        scores['h2'] += 1

    # Linguistic pattern scoring
    linguistic_score = linguistic_patterns.get('linguistic_score', 0)
    if linguistic_score > 5:
        scores['title'] += 1
        scores['h1'] += 1
        scores['h2'] += 1

    # Find best classification
    best_class = max(scores, key=lambda k: scores[k])
    confidence = scores[best_class] / sum(scores.values()) if sum(scores.values()) > 0 else 0

    return best_class, confidence

# MAIN ENHANCED PROCESSING FUNCTION
def process_pdf_enhanced(pdf_path: str) -> Dict:
    """Enhanced PDF processing with improved heuristics"""

    doc = pymupdf.open(pdf_path)
    df = extract_spans(doc)

    # Step 1: Global font analysis
    font_analysis = analyze_fonts_globally(df)

    # Step 2: Enhanced position features
    df = extract_position_features(df)

    # Step 3: Filter candidates with multiple criteria
    candidates = df[
        (df['size_norm'] >= 0.6) |  # Large font
        (df['is_bold']) |           # Bold text
        (df['text'].str.len() >= 3) # Minimum length
    ].copy()

    # Step 4: Enhanced classification
    results = []
    for idx, row in candidates.iterrows():
        # Linguistic analysis
        linguistic = analyze_linguistic_patterns(row['text'])

        # Context analysis
        context = analyze_context(df, idx) #type: ignore

        # Enhanced classification
        classification, confidence = classify_heading_enhanced(
            row, font_analysis, linguistic, context
        )

        if classification != 'none' and confidence > 0.3:
            results.append({
                'text': row['text'].strip(),
                'level': classification,
                'page': row['page'],
                'confidence': confidence,
                'font_size': row['size'],
                'is_bold': row['is_bold']
            })

    # Step 5: Post-processing and hierarchy validation
    results = sorted(results, key=lambda x: (x['page'], -x['font_size']))

    # Build final output
    title = ""
    outline = []

    for item in results:
        if item['level'] == 'title' and not title:
            title = item['text']
        elif item['level'] in ['h1', 'h2', 'h3']:
            outline.append({
                'level': item['level'].upper(),
                'text': item['text'],
                'page': item['page']
            })

    return {
        'title': title,
        'outline': outline[:50]  # Limit to prevent overload
    }
