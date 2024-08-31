import pandas as pd
import re
from bs4 import BeautifulSoup
import unicodedata
from tqdm import tqdm
from collections import Counter
import csv

def find_true_end(text, initial_end_pos, lookahead_range=1000):
    current_end_pos = initial_end_pos
    while True:
        lookahead_text = text[current_end_pos:current_end_pos + lookahead_range]
        next_end_match = re.search(r'([=]{5,}|[-]{5,}|[.]{5,}|-{10,})', lookahead_text)
        if next_end_match:
            current_end_pos += next_end_match.end()
        else:
            break
    return current_end_pos

def remove_table_from_text(text, stats):
    cleaned_text = ""
    position = 0
    removed_tables = []
    while True:
        start_match = re.search(r'\b[A-Z]{5,}\b', text[position:])
        if not start_match:
            cleaned_text += text[position:]
            break
        start_pos = position + start_match.start()
        cleaned_text += text[position:start_pos].strip() + "\n"
        lookahead_range = 500
        lookahead_text = text[start_pos:start_pos + lookahead_range]
        table_end_match = re.search(r'([=]{5,}|[-]{5,}|[.]{5,}|-{10,})', lookahead_text)
        if not table_end_match:
            position = start_pos + len(start_match.group(0))
            continue
        initial_end_pos = start_pos + table_end_match.end()
        true_end_pos = find_true_end(text, initial_end_pos)
        table_content = text[start_pos:true_end_pos]
        removed_tables.append(table_content)
        stats['tables_removed'] += 1
        position = true_end_pos
    return cleaned_text.strip(), removed_tables

def clean_text(text, useTable=True):
    stats = Counter()
    removed_tables = []
    
    # Count and remove HTML tags
    soup = BeautifulSoup(text, 'html.parser')
    stats['html_tags_removed'] = len(list(soup.find_all()))
    text = soup.get_text()
    
    # Remove specific citation patterns
    text, n = re.subn(r'\((?:\w+,?\s+(?:et al\.)?,?\s+)?(?:19|20)\d{2}[a-z]?(?::\d+(?:-\d+)?)?(?:\s+and\s+(?:\w+,?\s+(?:et al\.)?,?\s+)?(?:19|20)\d{2}[a-z]?(?::\d+(?:-\d+)?)?)*\)', '', text)
    stats['citations_removed'] += n
    
    # Remove content inside square brackets
    text, n = re.subn(r'\[.*?\]', '', text)
    stats['square_brackets_removed'] += n

    if useTable:
        # Table removal step
        text, tables = remove_table_from_text(text, stats)
        removed_tables.extend(tables)
    
    # Remove curly braces and content within them
    text, n = re.subn(r'\{.*?\}', '', text)
    
    # Remove only asterisks
    text, n = re.subn(r'\*+', '', text)
    
    # Remove table-like structures (leftover simple tables)
    text, n = re.subn(r'(?m)^\s*[\|+].*[\|+]\s*$', '', text)
    stats['table_like_structures_removed'] += n
    text, n = re.subn(r'(?m)^\s*[-+]+\s*$', '', text)
    stats['table_like_structures_removed'] += n
    
    # Remove equation patterns
    text, n = re.subn(r'(?m)^\s*[a-zA-Z0-9]+\s*[-+*/^()]+.*$', '', text)
    stats['equations_removed'] += n
    text, n = re.subn(r'(?m)^\s*[∑∫∏∂∇Δ].*$', '', text)
    stats['equations_removed'] += n
    
    # Remove equation patterns that use ()
    text, n = re.subn(r'\b[a-zA-Z0-9]+\s*[\+\-\*/\^]*\s*\(.*?\)\s*[\+\-\*/\^]*\s*[a-zA-Z0-9]*\b', '', text)
    stats['equations_with_parentheses_removed'] += n
    
    text, n = re.subn(r'(?m)^\s*[\(\)\[\]\{\}a-zA-Z0-9]+\s*[-+*/^()]+\s*\(.*?\)\s*.*$', '', text)
    stats['equations_with_parentheses_removed'] += n
    
    # Remove special characters commonly used in equations
    text, n = re.subn(r'[±∓×÷∙∘·°∂∇∆∑∏∫√∛∜∝∞≈≠≡≤≥≪≫⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋∈∉∋∌∍∎∏∐∑−]', '', text)
    stats['special_characters_removed'] += n
    
    # Remove sequences of numbers separated by one space
    text, n = re.subn(r'\b(\d+(?:\s+\d+)+)\b', '', text)
    stats['number_sequences_removed'] += n
    
    # Normalize dashes and hyphens
    text, n = re.subn(r'---+', '--', text)
    stats['dashes_normalized'] += n
    text, n = re.subn(r'[—–]', '-', text)
    stats['dashes_normalized'] += n
    
    # Normalize quotes and apostrophes
    text, n = re.subn(r'[""''""‹›«»]', "'", text)
    stats['quotes_normalized'] += n
    text, n = re.subn(r'[''´`]', "'", text)
    stats['apostrophes_normalized'] += n
    
    # Remove bullet points and list markers
    text, n = re.subn(r'[•◦▪▫▸▹►▻➤➢◆◇○●]', '', text)
    stats['bullet_points_removed'] += n
    
    # Remove URLs
    text, n = re.subn(r'http\S+|www\.\S+', '', text)
    stats['urls_removed'] += n
    
    # Remove email addresses
    text, n = re.subn(r'\S+@\S+', '', text)
    stats['email_addresses_removed'] += n
    
    # Remove footnote markers
    text, n = re.subn(r'(?<!\w)[\^\d+]', '', text)
    stats['footnote_markers_removed'] += n
    
    # Remove trademark symbols
    text, n = re.subn(r'[™®©℠]', '', text)
    stats['trademark_symbols_removed'] += n
    
    # Normalize fractions
    fraction_map = {
        '½': '1/2', '⅓': '1/3', '⅔': '2/3', '¼': '1/4', '¾': '3/4',
        '⅕': '1/5', '⅖': '2/5', '⅗': '3/5', '⅘': '4/5', '⅙': '1/6',
        '⅚': '5/6', '⅐': '1/7', '⅛': '1/8', '⅜': '3/8', '⅝': '5/8',
        '⅞': '7/8', '⅑': '1/9', '⅒': '1/10'
    }
    for frac, repl in fraction_map.items():
        text, n = re.subn(frac, repl, text)
        stats['fractions_normalized'] += n
    
    # Remove or normalize other Unicode characters
    original_length = len(text)
    text = ''.join(char for char in text if unicodedata.category(char)[0] != 'So')
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    stats['unicode_characters_removed'] = original_length - len(text)

    # Define specific words/phrases to remove with exact match, including capitalization
    words_to_remove = ["'Introduction", "'Summary", "'Abstract", "'Objective", "'Executive Summary", "'Aim:", "'Referral informationR", "'PART: EVALUATION", "' Introduction", "'SITUATION", "'AIM", "'INTRODUCTION", "'Project PlanningI", "') Selfish Genes and Group Selection", "'Part - A", "'a)", " b) ", " c) ", " d) ", " e) ", "(see figure ) ", "(Figure ) ", "( ", "FORMULA"]

    # Remove specified words/phrases from the text with exact case-sensitive matching
    for word in words_to_remove:
        text, n = re.subn(r'\b' + re.escape(word) + r'\b', '', text)
        stats[f'{word}_removed'] = n
    
    # Remove repeated punctuation
    text, n = re.subn(r'([!?.]){2,}', r'\1', text)
    stats['repeated_punctuation_removed'] += n
    
    # Normalize spaces around punctuation
    text, n = re.subn(r'\s+([,.!?:;])', r'\1', text)
    stats['spaces_normalized'] += n
    text, n = re.subn(r'([,.!?:;])\s+', r'\1 ', text)
    stats['spaces_normalized'] += n
    
    # Remove empty parentheses and patterns like :(a), :(b), :(c)
    text, n = re.subn(r'\(\s*\)', '', text)
    stats['empty_parentheses_removed'] += n
    text, n = re.subn(r'\(\s*[a-z]\s*\)', '', text)
    stats['single_letter_parentheses_removed'] += n

    # Remove fig in ()
    text, n = re.subn(r'\(\s*(Pl\.\s*\d+\s*,)?\s*Fig\.\s*\d+(\.\d+)?\s*\)', '', text)
    stats['figure_references_removed'] += n

    # Remove lines with excessive whitespace (potential tables)
    original_lines = text.split('\n')
    text = '\n'.join(line for line in original_lines if len(line.split()) > 1 or len(line.strip()) < 5)
    stats['excessive_whitespace_lines_removed'] = len(original_lines) - len(text.split('\n'))
    
    # Remove extra spaces and newlines
    original_length = len(text)
    text = re.sub(r'\s+', ' ', text).strip()
    stats['extra_spaces_removed'] = original_length - len(text)
    
    return text, stats, removed_tables