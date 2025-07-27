# ENHANCED PDF OUTLINE EXTRACTION - PERFORMANCE OPTIMIZED
# =======================================================
# This version replaces slow iterative processing with vectorized operations
# for significant speed improvements.

from utils import extract_spans
import pymupdf  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple

# 1. CROSS-PAGE FONT ANALYSIS - Unchanged
def analyze_fonts_globally(df: pd.DataFrame) -> Dict:
    """Analyze font patterns across the entire document for better consistency."""
    # Use simple clustering based on size gaps
    unique_sizes = sorted(df['size'].unique(), reverse=True)
    if not unique_sizes:
        return {'size_clusters': {}, 'font_frequency': {}, 'total_spans': len(df)}

    clusters = []
    current_cluster = [unique_sizes[0]]

    for i in range(1, len(unique_sizes)):
        # If size difference is small, group them
        if (current_cluster[0] - unique_sizes[i]) / current_cluster[0] < 0.1:
            current_cluster.append(unique_sizes[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [unique_sizes[i]]
    clusters.append(current_cluster)

    # Assign cluster rankings
    size_clusters = {size: rank for rank, cluster in enumerate(clusters, 1) for size in cluster}
    font_patterns = Counter(df['font'].values)

    return {
        'size_clusters': size_clusters,
        'font_frequency': dict(font_patterns),
        'total_spans': len(df)
    }

# 2. OPTIMIZED POSITION FEATURE EXTRACTION
def extract_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract position-based features from bounding boxes using vectorized operations."""
    df_enhanced = df.copy()

    # Extract position metrics from bbox [x0, y0, x1, y1]
    bboxes = df_enhanced['bbox'].tolist()
    df_enhanced['x0'] = [b[0] for b in bboxes]
    df_enhanced['y0'] = [b[1] for b in bboxes]
    df_enhanced['x1'] = [b[2] for b in bboxes]
    df_enhanced['y1'] = [b[3] for b in bboxes]

    # Position features
    df_enhanced['height'] = df_enhanced['y1'] - df_enhanced['y0']
    df_enhanced['left_margin'] = df_enhanced['x0']

    # Vectorized page-relative positioning using groupby().transform()
    page_groups = df_enhanced.groupby('page')
    
    # Normalize left margin
    page_min_margin = page_groups['left_margin'].transform('min')
    page_max_margin = page_groups['left_margin'].transform('max')
    df_enhanced['left_margin_norm'] = (df_enhanced['left_margin'] - page_min_margin) / \
                                      (page_max_margin - page_min_margin + 1e-6)

    return df_enhanced

# 3. VECTORIZED LINGUISTIC FEATURE EXTRACTION
def apply_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized creation of linguistic features."""
    text_stripped = df['text'].str.strip()
    
    df['has_section_pattern'] = text_stripped.str.match(r'^\d+\.\d*', na=False)
    df['has_numbers'] = text_stripped.str.contains(r'\d', na=False)
    df['all_caps'] = df['text'].str.isupper()
    df['title_case'] = df['text'].str.istitle()
    df['word_count'] = df['text'].str.split().str.len().fillna(0)
    df['ends_with_period'] = text_stripped.str.endswith('.', na=False)
    
    heading_keywords = {'introduction', 'conclusion', 'abstract', 'summary', 'background', 'methodology', 'results', 'discussion', 'chapter', 'section'}
    # This check is more complex, so apply is acceptable here if it's on a small subset or a single column.
    df['has_heading_keyword'] = df['text'].str.lower().str.split().apply(
        lambda words: any(word.strip(".,:") in heading_keywords for word in words) if isinstance(words, list) else False
    )

    # Calculate linguistic score
    score = (df['has_section_pattern'] * 3 +
             df['has_numbers'] * 1 +
             (df['title_case'] | df['all_caps']) * 2 +
             df['has_heading_keyword'] * 2 +
             (df['word_count'] <= 10) * 1 +
             (~df['ends_with_period']) * 1)
             
    df['linguistic_score'] = score
    return df

# 4. VECTORIZED CONTEXT FEATURE EXTRACTION
def apply_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized creation of context features."""
    # Ensure dataframe is sorted for correct gap calculation
    df_sorted = df.sort_values(['page', 'y0']).reset_index()

    # Get the bottom of the previous span on the same page
    df_sorted['prev_y1'] = df_sorted.groupby('page')['y1'].shift(1)
    
    # Calculate the vertical gap
    vertical_gap = df_sorted['y0'] - df_sorted['prev_y1']
    df_sorted['large_gap_before'] = vertical_gap > df_sorted['height'] * 1.5
    
    # The first element on each page has a large gap by definition
    df_sorted.loc[df_sorted.groupby('page').head(1).index, 'large_gap_before'] = True
    
    # Restore original index to merge back
    return df_sorted.set_index('index').sort_index()


# 5. MAIN PROCESSING FUNCTION (REWRITTEN FOR PERFORMANCE)
def process_pdf_enhanced(pdf_path: str) -> Dict:
    """
    Processes a PDF to extract a structured outline using vectorized operations
    for high performance.
    """
    doc = pymupdf.open(pdf_path)
    df = extract_spans(doc)

    if df.empty:
        return {'title': '', 'outline': []}

    # Step 1 & 2: Perform global analysis and add position features
    font_analysis = analyze_fonts_globally(df)
    df = extract_position_features(df)

    # Step 3: Add linguistic and context features in a vectorized manner
    df = apply_linguistic_features(df)
    df = apply_context_features(df)

    # Step 4: Classify all rows at once using the pre-computed features
    scores_df = pd.DataFrame(index=df.index)
    scores_df['title'] = 0
    scores_df['h1'] = 0
    scores_df['h2'] = 0
    scores_df['h3'] = 0
    
    # Font size scoring
    df['size_cluster'] = df['size'].map(font_analysis.get('size_clusters', {})).fillna(999)
    scores_df['title'] += (df['size_cluster'] == 1) & (df['page'] == 1) * 5
    scores_df['h1'] += ((df['size_cluster'] == 1) & (df['page'] != 1) * 5) + ((df['size_cluster'] == 2) * 3)
    scores_df['h2'] += ((df['size_cluster'] == 2) * 2) + ((df['size_cluster'] == 3) * 3)
    scores_df['h3'] += (df['size_cluster'] == 3) * 2

    # Style, position, and linguistic scoring
    scores_df['title'] += df['is_bold'] * 2 + (df['left_margin_norm'] < 0.1) * 1 + df['large_gap_before'] * 2
    scores_df['h1'] += df['is_bold'] * 2 + (df['left_margin_norm'] < 0.1) * 1 + df['large_gap_before'] * 2
    scores_df['h2'] += df['is_bold'] * 1 + df['large_gap_before'] * 1
    scores_df['h1'] += (df['linguistic_score'] > 5) * 1
    scores_df['h2'] += (df['linguistic_score'] > 5) * 1

    # Determine best class and confidence
    df['level'] = scores_df.idxmax(axis=1)
    total_scores = scores_df.sum(axis=1)
    df['confidence'] = scores_df.max(axis=1) / total_scores.replace(0, 1)

    # Filter out low-confidence and non-heading results
    headings = df[(df['confidence'] > 0.3) & (~df['level'].isin(['none', '']))].copy()
    
    # Step 5: Post-processing and building the final JSON
    headings = headings.sort_values(['page', 'y0'])
    
    # Simple title extraction: highest confidence 'title' on page 1
    title_candidates = headings[(headings['level'] == 'title') & (headings['page'] == 1)]
    title = title_candidates.loc[title_candidates['confidence'].idxmax()]['text'].strip() if not title_candidates.empty else ""

    # Build outline
    outline = []
    for _, row in headings.iterrows():
        level = row['level']
        if level in ['h1', 'h2', 'h3']:
            outline.append({
                'level': level.upper(),
                'text': row['text'].strip(),
                'page': int(row['page'])
            })

    return {
        'title': title,
        'outline': outline[:100]  # Limit to a reasonable number of headings
    }
