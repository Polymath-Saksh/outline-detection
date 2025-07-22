import pymupdf  # type: ignore[import]
import pandas as pd # type: ignore[import]
import json
import glob
import os

def load_pdf(pdf_path):
    return pymupdf.open(pdf_path)

def extract_spans(doc):
    spans_data = []
    for page_num, page in enumerate(doc, start=1):
        text_dict = page.get_text("dict")
        page_font_sizes = [
            span["size"]
            for block in text_dict["blocks"] if block["type"] == 0
            for line in block["lines"]
            for span in line["spans"]
        ]
        if page_font_sizes:
            max_font_size = max(page_font_sizes)
        else:
            max_font_size = 1.0
        for block in text_dict["blocks"]:
            if block["type"] == 0:
                for line in block["lines"]:
                    for span in line["spans"]:
                        flags = span.get("flags", 0)
                        if any(style in span["font"] for style in ["Bold", "Heading"]) or bool(flags & 2):
                            is_bold = True
                        else:
                            is_bold = False
                        is_italic = bool(flags & 1)
                        text = span["text"]
                        # Calculate capitalization ratio
                        num_upper = sum(1 for c in text if c.isupper())
                        num_alpha = sum(1 for c in text if c.isalpha())
                        capitalization_ratio = num_upper / num_alpha if num_alpha > 0 else 0
                        spans_data.append({
                            "page": page_num,
                            "text": text,
                            "font": span["font"],
                            "size": span["size"],
                            "bbox": span["bbox"],
                            "color": span["color"],
                            "flags": flags,
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "size_norm": span["size"] / max_font_size if max_font_size else 0,
                            "capitalization_ratio": capitalization_ratio
                        })
    return pd.DataFrame(spans_data)

def is_heading_like_font(font_name):
    return any(word in font_name for word in ["Bold", "Heading", "Black", "Heavy"])

def filter_outline_candidates(df):
    filtered_df = df[
        (df['size_norm'] >= 0.6) &
        (df['text'].str.strip().str.len() >= 3) &
        (df['is_bold'] | df['font'].apply(is_heading_like_font))
    ]
    return filtered_df.reset_index(drop=True)

def analyze_font_sizes(filtered_df):
    size_stats = filtered_df.groupby(['page', 'size']).size().reset_index(name='count')
    return size_stats.sort_values(['page', 'size'], ascending=[True, False])

def compute_size_ranks(filtered_df):
    size_order = (
        filtered_df.groupby('page')['size']
        .apply(lambda s: sorted(s.unique(), reverse=True))
        .to_dict()
    )
    size_ranks = {
        (page, size): rank+1
        for page, sizes in size_order.items()
        for rank, size in enumerate(sizes)
    }
    return size_ranks

def assign_outline_level(row, size_ranks):
    rank = size_ranks.get((row['page'], row['size']), None)
    if row['page'] == 1 and rank == 1:
        return 'title'
    elif rank == 1:
        return 'h1'
    elif rank == 2:
        return 'h2'
    elif rank == 3:
        return 'h3'
    else:
        return None

def add_outline_levels(filtered_df):
    size_ranks = compute_size_ranks(filtered_df)
    filtered_df['outline_level'] = filtered_df.apply(lambda row: assign_outline_level(row, size_ranks), axis=1)
    return filtered_df

def merge_outline_fragments(filtered_df):
    """
    Merge consecutive fragments for each outline level on the same page,
    but keep the original dataframe structure and attributes.
    Only keep fragments with <= 15 words.
    Ensures capitalization_ratio is always present in the output.
    """
    # Remove rows without an outline_level before sorting
    df = filtered_df[filtered_df['outline_level'].notnull()].copy()
    # Ensure capitalization_ratio exists in df
    if 'capitalization_ratio' not in df.columns:
        df['capitalization_ratio'] = 0.0
    merged_rows = []
    prev_level = prev_page = None
    buffer = ""
    buffer_row = None
    buffer_caps = []
    for idx, row in df.sort_values(['page', 'outline_level']).iterrows():
        level = row['outline_level']
        page = row['page']
        text = row['text'].strip()
        capitalization_ratio = row.get('capitalization_ratio', 0)
        if level is None or not text:
            continue
        if (level, page) == (prev_level, prev_page):
            buffer += " " + text
            buffer_caps.append(capitalization_ratio)
        else:
            if prev_level is not None and buffer_row is not None:
                if len(buffer.split()) <= 20:
                    new_row = buffer_row.copy()
                    new_row['text'] = buffer
                    # Use mean capitalization_ratio for merged fragment
                    if buffer_caps:
                        new_row['capitalization_ratio'] = sum(buffer_caps) / len(buffer_caps)
                    merged_rows.append(new_row)
            buffer = text
            buffer_row = row.copy()
            buffer_caps = [capitalization_ratio]
            prev_level = level
            prev_page = page
    # Add the last buffer
    if prev_level is not None and buffer_row is not None and buffer:
        if len(buffer.split()) <= 15:
            new_row = buffer_row.copy()
            new_row['text'] = buffer
            if buffer_caps:
                new_row['capitalization_ratio'] = sum(buffer_caps) / len(buffer_caps)
            merged_rows.append(new_row)
    # Return as DataFrame with original columns (including capitalization_ratio)
    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        # Ensure capitalization_ratio is present
        if 'capitalization_ratio' not in merged_df.columns:
            merged_df['capitalization_ratio'] = 0.0
        # Ensure columns order matches input
        merged_df = merged_df[df.columns]
        return merged_df
    else:
        # Ensure capitalization_ratio is present in empty DataFrame
        empty = df.iloc[0:0].copy()
        if 'capitalization_ratio' not in empty.columns:
            empty['capitalization_ratio'] = 0.0
        return empty

def build_outline_json_from_merged(merged_df):
    # Get title (first 'title' found) with <= 15 words
    title_row = merged_df[(merged_df['outline_level'] == 'title') & (merged_df['text'].str.split().str.len() <= 15)]
    title = title_row['text'].iloc[0] if not title_row.empty else ""
    # Map outline_level to H1/H2/H3, only if <= 15 words
    level_map = {'h1': 'H1', 'h2': 'H2', 'h3': 'H3'}
    outline = [
        {
            "level": level_map.get(row['outline_level']),
            "text": row['text'],
            "page": int(row['page'])
        }
        for _, row in merged_df.iterrows()
        if row['outline_level'] in level_map and len(row['text'].split()) <= 15
    ]
    return {
        "title": title,
        "outline": outline
    }