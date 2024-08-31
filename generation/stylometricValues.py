# Import necessary libraries
import stanza  # Stanford NLP library for tokenization and NLP tasks
import re  # Regular expressions for text processing
import math  # Mathematical functions, e.g., for Honoré's Statistic calculation
import logging
import numpy as np  # Importing numpy for variance calculation
import textstat
from readability import getmeasures
from collections import Counter, defaultdict
# Download and set up the Stanza pipeline (you can specify the language if needed)
stanza.download('en')  # Download the English model
# Initialize the Stanza pipeline with required processors
nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma,depparse,ner')
from wordfreq import word_frequency
from scipy.optimize import curve_fit  # For fitting the Zipfian model
from wordtangible import word_concreteness, avg_text_concreteness, concrete_abstract_ratio
import nltk
nltk.download('punkt_tab')
# Download CMU Pronouncing Dictionary if not already downloaded
import pronouncing

def process_text(text):
    """
    Processes the input text with the Stanza pipeline and returns the processed document.

    Parameters:
        text (str): The input text sample.

    Returns:
        doc: The processed document containing tokens, POS tags, and other NLP features.
    """
    return nlp(text)

def compute_herdans_v(doc):
    """
    Computes Herdan's V measure of lexical richness for the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The Herdan's V value.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    # Calculate the number of unique words (types)
    unique_words = len(set(words))
    
    if total_words == 0 or unique_words == 0:
        return 0
    
    # Calculate Herdan's V
    herdans_v = math.log(unique_words) / math.log(total_words)
    
    return herdans_v

def compute_brunets_w(doc):
    """
    Computes Brunet's W measure of lexical richness for the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The Brunet's W value.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    # Calculate the number of unique words (types)
    unique_words = len(set(words))
    
    # Edge case handling: avoid division by zero or invalid computations
    if total_words == 0 or unique_words == 0:
        return 0.0
    
    # Define the constant 'a'
    a = 0.165
    
    # Calculate Brunet's W
    brunets_w = total_words ** (unique_words ** (-a))
    
    return brunets_w

def tuldava_ln_ttr(doc):
    """
    Computes Tuldava's Log-Normal Type-Token Ratio (LN-TTR) for lexical richness in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: Tuldava's LN-TTR.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    # Calculate the number of unique words (types)
    unique_words = len(set(words))
    
    if total_words == 0 or unique_words == 0:
        return 0.0
    
    # Calculate Tuldava's LN-TTR
    ln_ttr = math.log(unique_words) / math.log(total_words)
    
    return ln_ttr

def simpsons_index(doc):
    """
    Computes Simpson's Index for lexical diversity in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: Simpson's Index (1 - D) for lexical diversity.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 1.0  # If there are no words, diversity is maximal (value of 1)

    # Calculate frequencies of each word
    word_freqs = Counter(words)
    
    # Calculate Simpson's Index (D)
    simpsons_d = sum((count / total_words) ** 2 for count in word_freqs.values())
    
    # Return 1 - D to reflect diversity (0 means high diversity, 1 means low diversity)
    return 1 - simpsons_d

def sichel_s_measure(doc):
    """
    Computes Sichel's S Measure for lexical diversity in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: Sichel's S Measure for lexical diversity.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0  # Return 0 if there are no words

    # Calculate the number of unique words (types)
    unique_words = len(set(words))
    
    if unique_words == 0:
        return 0  # Avoid division by zero if there are no unique words
    
    # Calculate Sichel's S Measure
    sichel_s = (unique_words - math.log10(unique_words)) / math.sqrt(total_words)
    
    return sichel_s

def orlov_sigma(doc):
    """
    Computes Orlov's Sigma for lexical diversity in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: Orlov's Sigma for lexical diversity.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0  # Return 0 if there are no words

    # Calculate frequencies of each word
    word_freqs = Counter(words)
    
    # Calculate the average frequency
    average_freq = total_words / len(word_freqs)
    
    # Calculate the sum of squared deviations from the average frequency
    sum_squared_deviations = sum((freq - average_freq) ** 2 for freq in word_freqs.values())
    
    # Calculate Orlov's Sigma
    orlov_sigma = math.sqrt(sum_squared_deviations / len(word_freqs))
    
    return orlov_sigma


# Define a standard set of POS tags
STANDARD_POS_TAGS = [
    'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
    'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
]

def pos_frequencies(doc):
    """
    Computes the frequencies of different part-of-speech tags in the given processed document.
    Parameters:
        doc: The processed document from the Stanza pipeline.
    Returns:
        dict: A dictionary with standardized POS tags as keys and their frequencies as values.
    """
    # Extract POS tags from the processed document
    pos_tags = [word.upos for sentence in doc.sentences for word in sentence.words]
    
    # Calculate frequencies of each POS tag
    pos_freqs = Counter(pos_tags)
    
    # Convert frequencies to proportions
    total_pos_tags = len(pos_tags)
    pos_frequencies = {tag: pos_freqs.get(tag, 0) / total_pos_tags for tag in STANDARD_POS_TAGS}
    
    # Create the final dictionary with the 'pos_frequencies_' prefix
    return {f'pos_frequencies_{tag}': freq for tag, freq in pos_frequencies.items()}



def clauses_per_sentence(doc):
    """
    Computes the average number of clauses per sentence in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The average number of clauses per sentence.
    """
    ##print(type(doc))

    total_clauses = 0
    total_sentences = len(doc.sentences)
    
    if total_sentences == 0:
        return 0  # Avoid division by zero if there are no sentences

    # Iterate over sentences
    for sentence in doc.sentences:
        # Count clauses
        # Considering dependency relations indicative of clauses
        clause_relations = {'acl', 'ccomp', 'xcomp', 'advcl', 'relcl', 'nmod'}
        num_clauses = sum(word.deprel in clause_relations for word in sentence.words)
        
        total_clauses += num_clauses
    
    # Calculate average clauses per sentence
    avg_clauses_per_sentence = total_clauses / total_sentences
    
    return avg_clauses_per_sentence

def modifiers_per_noun_phrase(doc):
    """
    Computes the average number of modifiers per noun phrase in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The average number of modifiers per noun phrase.
    """
    total_modifiers = 0
    total_noun_phrases = 0
    sentence_modifiers = []  # List to record the number of modifiers per sentence
    
    # Define modifier relations
    modifier_relations = {'amod', 'nmod', 'acl', 'advmod', 'det', 'appos'}
    
    # Iterate over sentences
    for sentence in doc.sentences:
        noun_phrases = [word for word in sentence.words if word.upos == 'NOUN' or word.upos == 'PROPN']
        
        if not noun_phrases:
            ##print(f"DEBUG: No noun phrases found in sentence: '{sentence.text}'")
            continue
        
        total_noun_phrases += len(noun_phrases)
        sentence_mod_count = 0  # Counter for modifiers in the current sentence
        
        # Count modifiers for each noun phrase
        for noun_phrase in noun_phrases:
            modifiers = [word for word in sentence.words if word.deprel in modifier_relations and word.head == noun_phrase.id]
            ##print(f"DEBUG: Noun phrase '{noun_phrase.text}' has {len(modifiers)} modifiers.")
            total_modifiers += len(modifiers)
            sentence_mod_count += len(modifiers)
        
        sentence_modifiers.append(sentence_mod_count)
        ##print(f"DEBUG: Sentence '{sentence.text}' has {sentence_mod_count} total modifiers.")
    
    # Calculate average modifiers per noun phrase
    if total_noun_phrases == 0:
        ##print("DEBUG: No noun phrases found in the entire document.")
        return 0  # Avoid division by zero if there are no noun phrases

    avg_modifiers_per_noun_phrase = total_modifiers / total_noun_phrases
    
    ##print(f"DEBUG: Total noun phrases = {total_noun_phrases}, Total modifiers = {total_modifiers}, Average = {avg_modifiers_per_noun_phrase}")
    ##print(f"DEBUG: Modifiers per sentence: {sentence_modifiers}")
    
    return avg_modifiers_per_noun_phrase

def coordinated_phrases_per_sentence(doc):
    """
    Computes the average number of coordinated phrases per sentence in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The average number of coordinated phrases per sentence.
    """
    total_coordinated_phrases = 0
    total_sentences = len(doc.sentences)
    
    if total_sentences == 0:
        return 0  # Avoid division by zero if there are no sentences

    # Iterate over sentences
    for sentence in doc.sentences:
        # Extract the list of words and their dependencies
        words = sentence.words
        
        # List of coordinating conjunctions
        coordinating_conjunctions = {'cc'}
        
        # Find coordinated phrases based on conjunctions
        coordinated_phrases = 0
        for word in words:
            if word.deprel in coordinating_conjunctions:
                # Find the number of phrases coordinated by this conjunction
                # The conjunction typically connects two or more phrases
                coordinated_phrases += 1
        
        total_coordinated_phrases += coordinated_phrases
    
    # Calculate average coordinated phrases per sentence
    avg_coordinated_phrases_per_sentence = total_coordinated_phrases / total_sentences
    
    return avg_coordinated_phrases_per_sentence

def sentence_length_variation(doc):
    """
    Computes the variation in sentence lengths in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The standard deviation of sentence lengths.
    """
    sentence_lengths = [len(sentence.words) for sentence in doc.sentences]
    
    if not sentence_lengths:
        return 0  # Return 0 if there are no sentences

    # Calculate the standard deviation of sentence lengths
    std_dev = np.std(sentence_lengths)

    return std_dev

def maas_index(doc):
    """
    Computes Maas' Index for lexical richness in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: Maas' Index.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    # Calculate the number of unique words (types)
    unique_words = len(set(words))
    
    if total_words == 0 or unique_words == 0:
        return 0.0
    
    # Calculate Maas' Index
    maas_index_value = (math.log(total_words) - math.log(unique_words)) / (math.log(total_words) ** 2)
    
    return maas_index_value

def extract_clauses(doc):
    """
    Extracts clauses from a Stanza Document object based on dependency relations.
    Parameters:
    doc (Document): A Stanza Document object.
    Returns:
    list of lists: A list where each sublist represents a clause of Word objects.
    """
    clauses = []
    for sentence in doc.sentences:
        current_clause = []
        for word in sentence.words:
            if word.deprel in {'ccomp', 'xcomp', 'acl', 'advcl', 'relcl'}:
                if current_clause:
                    clauses.append(current_clause)
                current_clause = []
            current_clause.append(word)
        if current_clause:
            clauses.append(current_clause)
    return clauses

def clause_length_variation(doc):
    """
    Computes the variation in clause lengths in the given processed document.
    Parameters:
    doc: The processed document from the Stanza pipeline.
    Returns:
    float: The standard deviation of clause lengths.
    """
    clauses = extract_clauses(doc)
    clause_lengths = [len(clause) for clause in clauses]
    
    if not clause_lengths:
        return 0  # Return 0 if there are no clauses
    
    # Calculate the standard deviation of clause lengths
    std_dev = np.std(clause_lengths)
    return std_dev

def subordination_index(doc):
    """
    Computes the Subordination Index for the given document.
    Parameters:
    doc (Document): The Stanza Document object.
    Returns:
    float: The Subordination Index.
    """
    clauses = extract_clauses(doc)
    total_clauses = len(clauses)
    subordinate_clauses = sum(1 for clause in clauses if any(word.deprel in {'ccomp', 'xcomp', 'acl', 'advcl', 'relcl'} for word in clause))
    
    if total_clauses == 0:
        return 0
    
    # Calculate Subordination Index
    subordination_index = subordinate_clauses / total_clauses
    return subordination_index

def compute_sentence_depth(sentence):
    """
    Computes the depth of a single sentence's parse tree.
    Parameters:
    sentence (stanza.Sentence): A Stanza Sentence object.
    Returns:
    int: The depth of the sentence's parse tree.
    """
    def get_depth(word_id, depth_map):
        if word_id in depth_map:
            return depth_map[word_id]
        
        word = sentence.words[word_id - 1]  # Stanza uses 1-based indexing
        if word.head == 0:  # Root node
            depth = 0
        else:
            depth = 1 + get_depth(word.head, depth_map)
        
        depth_map[word_id] = depth
        return depth

    depth_map = {}
    max_depth = 0
    for word in sentence.words:
        depth = get_depth(word.id, depth_map)
        max_depth = max(max_depth, depth)

    return max_depth

def average_sentence_depth(doc):
    """
    Computes the average sentence depth for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The average depth of sentences in the text.
    """
    depths = []
    
    for sentence in doc.sentences:
        depth = compute_sentence_depth(sentence)
        depths.append(depth)
    
    # Calculate average depth
    if len(depths) == 0:
        return 0
    
    average_depth = sum(depths) / len(depths)
    
    return average_depth

def extract_coordinate_phrases(doc):
    """
    Extracts coordinate phrases from a processed document.
    
    Parameters:
        doc: The processed document from the Stanza pipeline.
    
    Returns:
        tuple: A tuple containing:
            - coordinate_phrases: List of coordinate phrases.
            - total_phrases: List of all phrases.
    """
    coordinate_phrases = []
    total_phrases = []
    coord_conjunctions = {'and', 'or', 'but', 'nor', 'for', 'so', 'yet'}
    
    current_phrase = []
    
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.text.lower() in coord_conjunctions:
                if current_phrase:
                    total_phrases.append(' '.join(current_phrase))
                    current_phrase = []
                coordinate_phrases.append(word.text)
            else:
                current_phrase.append(word.text)
        
        if current_phrase:
            total_phrases.append(' '.join(current_phrase))
            current_phrase = []
    
    return coordinate_phrases, total_phrases

def coordinate_phrases_ratio(doc):
    """
    Computes the Coordinate Phrases Ratio for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The Coordinate Phrases Ratio.
    """
    coordinate_phrases, total_phrases = extract_coordinate_phrases(doc)

    num_coordinate_phrases = len(coordinate_phrases)
    num_total_phrases = len(total_phrases)
    
    if num_total_phrases == 0:
        return 0
    
    # Calculate Coordinate Phrases Ratio
    ratio = num_coordinate_phrases / num_total_phrases
    
    return ratio

def extract_dependent_clauses(doc):
    """
    Extracts dependent clauses from a processed document.
    
    Parameters:
        doc: The processed document from the Stanza pipeline.
    
    Returns:
        list: A list of dependent clauses.
    """
    dependent_clauses = []
    dep_clause_rels = {'ccomp', 'xcomp', 'acl', 'advcl', 'relcl', 'mark'}

    current_clause = []
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel in dep_clause_rels:
                if current_clause:
                    dependent_clauses.append(' '.join(current_clause))
                    current_clause = []
                dependent_clauses.append(word.text)
            else:
                current_clause.append(word.text)
        
        if current_clause:
            dependent_clauses.append(' '.join(current_clause))
            current_clause = []

    return dependent_clauses

def dependent_clauses_ratio(doc):
    """
    Computes the Dependent Clauses Ratio for the given text.
    
    Parameters:
        text (str): The input text.
    
    Returns:
        float: The Dependent Clauses Ratio.
    """
    
    # Extract dependent clauses
    dependent_clauses = extract_dependent_clauses(doc)
    
    # Calculate total number of clauses
    total_clauses = len(dependent_clauses)  # Since we are only extracting dependent clauses
    
    # Calculate total number of clauses by considering both dependent and independent clauses
    all_clauses = [sentence.text for sentence in doc.sentences]
    num_total_clauses = len(all_clauses)
    
    if num_total_clauses == 0:
        return 0
    
    # Calculate Dependent Clauses Ratio
    ratio = len(dependent_clauses) / num_total_clauses
    
    return ratio

def dugast_vmi(doc):
    """
    Computes Dugast's Vocabulary Management Index (VMI) for the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: Dugast's Vocabulary Management Index (VMI).
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    # Calculate the number of unique words (types)
    unique_words = len(set(words))
    
    if total_words == 0:
        return 0.0
    
    # Calculate Dugast's Vocabulary Management Index (VMI)
    vmi = (unique_words ** 2) / total_words
    
    return vmi

def yules_k_characteristic(doc):
    """
    Computes Yule's K characteristic measure of lexical richness for the given processed document.
    Parameters:
    doc: The processed document from the Stanza pipeline.
    Returns:
    float: The Yule's K characteristic value.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    if total_words == 0:
        ##print("DEBUG: No words found in the document.")
        return 0
    
    # Calculate frequency of each word
    frequency_distribution = {}
    for word in words:
        frequency_distribution[word] = frequency_distribution.get(word, 0) + 1
    
    # Calculate M1 (total number of words)
    M1 = total_words
    
    # Calculate M2 (sum of the squares of the frequencies)
    M2 = sum(freq ** 2 for freq in frequency_distribution.values())
    
    # Debug ##print to check intermediate values
    ##print(f"DEBUG: M1 = {M1}, M2 = {M2}, Unique words = {len(frequency_distribution)}")
    
    # Calculate Yule's K
    if M1 == 0:
        return 0  # Avoid division by zero
    
    yules_k = 10000 * (M2 - M1) / (M1 ** 2)
    
    # Ensure the result is non-negative
    yules_k = max(0, yules_k)
    
    ##print(f"DEBUG: Yule's K = {yules_k}")
    return yules_k



def honores_r(doc):
    """
    Computes Honoré's R statistic for lexical richness for the given processed document.
    Parameters:
        doc: The processed document from the Stanza pipeline.
    Returns:
        float: The Honoré's R value, or 0.0 if the calculation is not possible.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    # Calculate the number of unique words (types)
    unique_words = len(set(words))
    
    # Calculate V1 (the number of words that occur exactly once)
    hapax_legomena = sum(1 for word in set(words) if words.count(word) == 1)
    
    # Log debugging information
    logging.debug(f"Total words: {total_words}, Unique words: {unique_words}, Hapax legomena: {hapax_legomena}")
    
    # Edge case handling: avoid division by zero or invalid computations
    if total_words == 0 or unique_words == 0:
        logging.warning("Honore's R calculation failed: total_words or unique_words is 0")
        return 0.0
    
    if hapax_legomena == unique_words:
        logging.warning("Honore's R calculation failed: all unique words are hapax legomena")
        return 0.0
    
    try:
        # Calculate Honoré's R
        honores_r_value = 100 * (math.log(total_words)) / (1 - (hapax_legomena / unique_words))
        
        # Check for invalid results
        if math.isnan(honores_r_value) or math.isinf(honores_r_value):
            logging.warning(f"Honore's R calculation resulted in an invalid value: {honores_r_value}")
            return 0.0
        
        return honores_r_value
    except Exception as e:
        logging.error(f"Error in Honore's R calculation: {str(e)}")
        return 0.0




def renyis_entropy(doc, alpha=2):
    """
    Computes Rényi's entropy of order alpha for lexical richness of the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.
        alpha (float): The order of Rényi's entropy (default is 2).

    Returns:
        float: The Rényi's entropy value.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate the total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Calculate frequency distribution of words
    frequency_distribution = {}
    for word in words:
        if word in frequency_distribution:
            frequency_distribution[word] += 1
        else:
            frequency_distribution[word] = 1
    
    # Calculate the probability of each word
    probabilities = [freq / total_words for freq in frequency_distribution.values()]
    
    # Calculate Rényi's entropy
    if alpha == 1:
        # Special case: alpha = 1, use Shannon's entropy formula
        entropy = -sum(p * math.log2(p) for p in probabilities)
    else:
        entropy_sum = sum(p ** alpha for p in probabilities)
        entropy = (1 / (1 - alpha)) * math.log2(entropy_sum)
    
    return entropy

def hapax_dislegomena_rate(doc):
    """
    Computes the Hapax Dislegomena Rate for the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The Hapax Dislegomena Rate.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Calculate the frequency of each word
    frequency_distribution = {}
    for word in words:
        if word in frequency_distribution:
            frequency_distribution[word] += 1
        else:
            frequency_distribution[word] = 1
    
    # Calculate V2 (the number of words that occur exactly twice)
    hapax_dislegomena = sum(1 for freq in frequency_distribution.values() if freq == 2)
    
    # Calculate Hapax Dislegomena Rate
    hapax_dislegomena_rate = hapax_dislegomena / total_words
    
    return hapax_dislegomena_rate

def perplexity(doc):
    """
    Computes the perplexity of a given text based on its word frequency distribution.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The perplexity value.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return float('inf')  # Perplexity is undefined for empty text
    
    # Calculate frequency distribution of words
    frequency_distribution = {}
    for word in words:
        if word in frequency_distribution:
            frequency_distribution[word] += 1
        else:
            frequency_distribution[word] = 1
    
    # Calculate probabilities of each word
    probabilities = [freq / total_words for freq in frequency_distribution.values()]
    
    # Calculate perplexity
    perplexity = 2 ** (-sum(p * math.log2(p) for p in probabilities) / total_words)
    
    return perplexity

def burstiness(doc, num_segments=10):
    """
    Computes the burstiness of word occurrences in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.
        num_segments (int): The number of segments to divide the text into.

    Returns:
        float: The average burstiness of the words in the text.
    """
    # Extract words from the processed document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Determine the segment length
    segment_length = total_words // num_segments
    if segment_length == 0:
        return 0.0  # Too few words to segment effectively
    
    # Segment the text and count word occurrences in each segment
    segments = [words[i:i + segment_length] for i in range(0, total_words, segment_length)]
    
    # Calculate burstiness for each word
    burstiness_values = []
    for word in set(words):
        counts = [segment.count(word) for segment in segments]
        if np.mean(counts) > 0:  # Avoid division by zero
            burstiness_values.append(np.var(counts) / np.mean(counts))
    
    # Calculate the average burstiness across all words
    average_burstiness = np.mean(burstiness_values) if burstiness_values else 0.0
    
    return average_burstiness

def average_word_length(doc):
    """
    Computes the average word length in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The average word length.
    """
    # Extract words from the processed document
    words = [word.text for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Calculate the total length of all words
    total_length = sum(len(word) for word in words)
    
    # Calculate the average word length
    average_length = total_length / total_words
    
    return average_length

def long_word_rate(doc, length_threshold=7):
    """
    Computes the rate of long words (words with length >= length_threshold) in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.
        length_threshold (int): The minimum length for a word to be considered "long" (default is 7).

    Returns:
        float: The long word rate.
    """
    # Extract words from the processed document
    words = [word.text for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Calculate the number of long words
    long_words_count = sum(1 for word in words if len(word) >= length_threshold)
    
    # Calculate the long word rate
    long_word_rate = long_words_count / total_words
    
    return long_word_rate

def short_word_rate(doc, length_threshold=4):
    """
    Computes the rate of short words (words with length < length_threshold) in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.
        length_threshold (int): The maximum length for a word to be considered "short" (default is 4).

    Returns:
        float: The short word rate.
    """
    # Extract words from the processed document
    words = [word.text for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Calculate the number of short words
    short_words_count = sum(1 for word in words if len(word) < length_threshold)
    
    # Calculate the short word rate
    short_word_rate = short_words_count / total_words
    
    return short_word_rate

def lexical_density(doc):
    """
    Computes the lexical density of the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The lexical density of the text.
    """
    # Extract words and their POS tags from the processed document
    content_word_tags = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    
    total_words = 0
    content_words = 0
    
    for sentence in doc.sentences:
        for word in sentence.words:
            total_words += 1
            if word.upos in content_word_tags:
                content_words += 1
    
    if total_words == 0:
        return 0.0
    
    # Calculate lexical density
    lexical_density = content_words / total_words
    
    return lexical_density

def is_state_description(sentence):
    copular_verbs = {'be', 'seem', 'appear', 'look', 'sound', 'smell', 'taste', 'feel', 'become', 'get'}
    state_adjectives = {'done', 'finished', 'completed', 'solved', 'gone', 'ready'}
    
    return any(word.lemma.lower() in copular_verbs and word.deprel == 'cop' for word in sentence.words) and \
           any(word.lemma.lower() in state_adjectives for word in sentence.words)

def passive_confidence(sentence):
    score = 0
    has_aux_pass = any(word.deprel == 'aux:pass' for word in sentence.words)
    has_nsubj_pass = any(word.deprel == 'nsubj:pass' for word in sentence.words)
    
    if has_aux_pass:
        score += 0.4
    if has_nsubj_pass:
        score += 0.4
    
    has_vbn = any(
        word.upos == 'VERB' and 
        word.feats and 'VerbForm=Part' in word.feats and
        not (word.deprel == 'amod' or  
             (word.deprel == 'root' and not has_aux_pass and not has_nsubj_pass))
        for word in sentence.words
    )
    if has_vbn:
        score += 0.2
    
    copular_verbs = {'be', 'seem', 'appear', 'look', 'sound', 'smell', 'taste', 'feel', 'become', 'get'}
    is_copular = any(word.lemma.lower() in copular_verbs and word.deprel == 'cop' for word in sentence.words)
    if is_copular:
        score -= 0.2
    
    # Semantic analysis
    state_verbs = {'solve', 'finish', 'complete', 'do', 'decide', 'resolve', 'settle', 'end', 'conclude'}
    action_verbs = {'chase', 'write', 'bake', 'sign', 'make', 'believe', 'paint', 'postpone', 
                    'elect', 'publish', 'destroy', 'expect', 'submit', 'found', 'attend'}
    
    main_verb = next((word for word in sentence.words if word.upos == 'VERB' and word.deprel == 'root'), None)
    
    if main_verb:
        if main_verb.lemma in state_verbs:
            score -= 0.2
        elif main_verb.lemma in action_verbs:
            score += 0.1
    
    # Check for 'get' passive
    if any(word.lemma == 'get' and word.deprel == 'aux' and 
           any(w.feats and 'VerbForm=Part' in w.feats for w in sentence.words) 
           for word in sentence.words):
        score += 0.2
    
    # Check for 'have been' passive
    if any(word.lemma == 'have' and word.deprel == 'aux' and
           any(w.lemma == 'be' for w in sentence.words) and
           any(w.feats and 'VerbForm=Part' in w.feats for w in sentence.words)
           for word in sentence.words):
        score += 0.2
    
    # Increase score for progressive passive
    if any(word.lemma == 'be' and word.deprel == 'aux' and 
           any(w.lemma == 'be' and w.feats and 'VerbForm=Part' in w.feats for w in sentence.words) 
           for word in sentence.words):
        score += 0.1

    # Increase score for impersonal passive
    if any(word.lemma == 'it' and word.deprel == 'expl' and
           any(w.deprel == 'aux:pass' for w in sentence.words)
           for word in sentence.words):
        score += 0.1
    
    # Reduce score for likely state descriptions
    if is_state_description(sentence):
        score -= 0.3
    
    return max(0, min(score, 1.0))

def analyze_passiveness(doc):
    confidence_scores = [passive_confidence(sentence) for sentence in doc.sentences]
    
    avg_passiveness = np.mean(confidence_scores)
    std_passiveness = np.std(confidence_scores)
    
    return {
        "average_passiveness": avg_passiveness,
        "std_passiveness": std_passiveness
    }

def is_cleft_sentence(sentence):
    words = sentence.words
    text = ' '.join(word.text.lower() for word in words)
    
    # Exclude certain patterns that are not cleft sentences
    exclude_patterns = [
        r'\bit\s+(is|was)\s+(obvious|clear|evident|apparent|known|understood)\s+that\b',
        r'\bit\s+seems\s+that\b'
    ]
    if any(re.search(pattern, text) for pattern in exclude_patterns):
        return False
    
    # Detect 'it' clefts
    if words[0].text.lower() == 'it':
        cop_found = False
        for i, word in enumerate(words[1:], start=1):
            if word.deprel == 'cop' and word.pos == 'AUX':
                cop_found = True
            elif cop_found and word.pos in ['PRON', 'SCONJ'] and word.text.lower() in ['who', 'whom', 'that', 'which', 'when', 'where', 'why', 'how']:
                return True
        if cop_found and re.search(r'\bit\s+(is|was)\s+\w+\s+(that|who|whom|which|when|where|why|how)', text):
            return True
    
    # Detect pseudo-clefts and wh-clefts
    wh_words = ['what', 'who', 'whom', 'which', 'where', 'when', 'why', 'how']
    if words[0].text.lower() in wh_words:
        for i, word in enumerate(words[1:], start=1):
            if word.deprel == 'cop' and word.pos == 'AUX':
                return True
            if i < len(words) - 1 and words[i+1].deprel in ['nsubj', 'ccomp', 'xcomp', 'advcl']:
                return True

    # Check for reverse pseudo-clefts and patterns that could indicate cleft sentences
    if any(re.search(rf'\b{wh}\s+.+?\s+(is|was|are|were)\s+.+\b', text) for wh in wh_words):
        return True

    # Specific pattern checks to capture edge cases
    patterns = [
        r'\bwhat\s+.+?\s+(is|was|are|were)\b',
        r'\bwhere\s+\w+\s+(is|was|are|were)\b',
        r'\bwhen\s+\w+\s+(is|was|are|were)\b',
        r'\bwhy\s+\w+\s+(is|was|are|were)\b',
        r'\bhow\s+\w+\s+(is|was|are|were)\b',
        r'\bwho\s+\w+\s+(is|was|are|were)\b',
        r'\bwhom\s+\w+\s+(is|was|are|were)\b',
        r'\bwhich\s+\w+\s+(is|was|are|were)\b',
        r'\b(is|was|are|were)\s+(what|where|when|why|how|who|whom|which)\b',
        r'\bthe\s+\w+\s+(who|that|which)\s+\w+\s+(is|was|are|were)\b',
        r'\bit\s+(is|was)\s+\w+\s+(when|where|why|how)\b',
        r'\bthe\s+\w+\s+(when|where)\s+\w+\s+(is|was|are|were)\b',
        r'\bwhat\s+\w+\s+(is|was)\s+.+\b',
        r'\b(is|was)\s+the\s+\w+\s+that\s+.+\b',
        r'\bwho\s+\w+\s+\w+\s+.+\b',
        r'\bwhat\s+\w+\s+\w+\s+.+\b',
        r'\bthe\s+\w+\s+is\s+\w+\s+\w+\s+.+\b',
        r'\bthe\s+\w+\s+that\s+.+?\s+is\b',
        r'\ball\s+\w+\s+\w+\s+.+?\s+is\b'
    ]
    if any(re.search(pattern, text) for pattern in patterns):
        return True
    
    return False

def ratio_of_cleft_sentences(doc):
    """
    Computes the ratio of cleft sentences in the given document.

    Parameters:
        doc (stanza.Document): The input document.

    Returns:
        float: The ratio of cleft sentences in the document.
    """
    sentences = doc.sentences
    
    if not sentences:
        return 0.0
    
    cleft_count = sum(1 for sentence in sentences if is_cleft_sentence(sentence))
    
    ratio = cleft_count / len(sentences)
    
    return ratio

def count_assonance(text):
    """
    Counts occurrences of assonance in the given text using phonetic analysis.
    
    Parameters:
        text (str): The input text.
    
    Returns:
        int: The number of assonance occurrences.
    """
    vowels = 'aeiou'
    assonance_count = 0
    words = text.split()
    
    # Get the phonetic representation of each word
    phonetic_words = {word: pronouncing.phones_for_word(word) for word in words}
    
    for i in range(len(words) - 1):
        word1, word2 = words[i].lower(), words[i + 1].lower()
        
        # Get phonetic representations
        phones1 = phonetic_words.get(word1, [])
        phones2 = phonetic_words.get(word2, [])
        
        # Check for common vowel sounds
        for phone1 in phones1:
            for phone2 in phones2:
                if any(vowel in phone1 and vowel in phone2 for vowel in vowels):
                    assonance_count += 1
                    break
    
    return assonance_count

def normalized_assonance(text):
    """
    Computes the normalized assonance score based on the length of the text.
    
    Parameters:
        text (str): The input text.
    
    Returns:
        float: The normalized assonance score (assonance count per word).
    """
    assonance_count = count_assonance(text)
    num_words = len(text.split())
    
    return assonance_count / num_words if num_words > 0 else 0.0

def count_alliteration(text):
    """
    Counts occurrences of alliteration in the given text using phonetic analysis.
    
    Parameters:
        text (str): The input text.
    
    Returns:
        int: The number of alliteration occurrences.
    """
    alliteration_count = 0
    words = text.split()
    
    # Get the phonetic representation of each word
    phonetic_words = {word: pronouncing.phones_for_word(word) for word in words}
    
    for i in range(len(words) - 1):
        initial1 = get_initial_consonant(words[i])
        initial2 = get_initial_consonant(words[i + 1])
        
        if initial1 and initial2 and initial1 == initial2:
            alliteration_count += 1
    
    return alliteration_count

def get_initial_consonant(word):
    """
    Extracts the initial consonant sound of a word using its phonetic representation.
    
    Parameters:
        word (str): The input word.
    
    Returns:
        str: The initial consonant sound.
    """
    phones = pronouncing.phones_for_word(word)
    if phones:
        phonetic_rep = phones[0]
        match = re.match(r'[^aeiou\W]*', phonetic_rep)
        return match.group(0) if match else ''
    return ''

def normalized_alliteration(text):
    """
    Computes the normalized alliteration score based on the length of the text.
    
    Parameters:
        text (str): The input text.
    
    Returns:
        float: The normalized alliteration score (alliteration count per word).
    """
    alliteration_count = count_alliteration(text)
    num_words = len(text.split())
    
    return alliteration_count / num_words if num_words > 0 else 0.0

def tempo_variation(doc):
    """
    Computes the tempo variation based on syllable counts per sentence in the document.
    
    Parameters:
        doc (stanza.Document): The Stanza Document object.

    Returns:
        float: The standard deviation of syllable counts per sentence.
    """
    syllable_counts_per_sentence = []

    for sentence in doc.sentences:
        syllable_count = sum(len(pronouncing.phones_for_word(word.text)) for word in sentence.words if pronouncing.phones_for_word(word.text))
        syllable_counts_per_sentence.append(syllable_count)

    if len(syllable_counts_per_sentence) > 1:
        return np.std(syllable_counts_per_sentence)
    return 0.0

def rhythmic_complexity(text):
    """
    Computes the rhythmic complexity based on stress patterns.
    
    Parameters:
        text (str): The input text.

    Returns:
        float: The standard deviation of stress pattern counts.
    """
    words = text.split()
    stress_patterns = []

    def get_stress_pattern(word):
        phones = pronouncing.phones_for_word(word)
        if phones:
            stress_pattern = pronouncing.stresses(phones[0])
            return [int(s == '1') for s in stress_pattern]
        return []

    for word in words:
        stress_patterns.extend(get_stress_pattern(word))

    if len(stress_patterns) > 1:
        return np.std(stress_patterns)
    return 0.0

def prosodic_patterns(doc):
    """
    Computes prosodic patterns by analyzing sentence lengths.
    
    Parameters:
        doc (stanza.Document): The Stanza Document object.

    Returns:
        float: The standard deviation of sentence lengths.
    """
    sentence_lengths = []

    for sentence in doc.sentences:
        sentence_length = len(sentence.words)
        sentence_lengths.append(sentence_length)

    if len(sentence_lengths) > 1:
        return np.std(sentence_lengths)
    return 0.0

def is_fronted_adverbial(word, sentence, pos_tags=None, dep_rels=None, clause_boundary_tags=None):
    """
    Check if a word is a fronted adverbial with configurable POS tags and dependency relations,
    considering multi-token phrases and advanced clause analysis.
    """
    # Default POS tags and dependency relations if none are provided
    if pos_tags is None:
        pos_tags = {"VERB", "AUX"}
    if dep_rels is None:
        dep_rels = {"advmod", "nmod", "obl", "prep", "mark", "acl", "advcl", "xcomp"}
    if clause_boundary_tags is None:
        clause_boundary_tags = {"CCONJ", "PUNCT"}  # Conjunctions and punctuation that may mark clause boundaries

    # Ensure the word is at or near the start of the sentence
    if word.id == 1 or word.id == 2:
        # Check if it's an adverbial modifier with the appropriate head POS tag
        if hasattr(word, 'deprel') and hasattr(word, 'head'):
            head_word = sentence[word.head - 1] if isinstance(word.head, int) and 0 < word.head <= len(sentence) else None
            if word.deprel in dep_rels and head_word and hasattr(head_word, 'upos') and head_word.upos in pos_tags:
                return True

    # Advanced parsing: Check for adverbials that may start after an initial clause boundary
    for i, w in enumerate(sentence[:5]):  # Limiting to the first five words for efficiency
        if hasattr(w, 'deprel') and hasattr(w, 'head'):
            head_word = sentence[w.head - 1] if isinstance(w.head, int) and 0 < w.head <= len(sentence) else None
            if w.deprel in dep_rels and head_word and hasattr(head_word, 'upos') and head_word.upos in pos_tags:
                if i == 0 or (i > 0 and hasattr(sentence[i-1], 'upos') and sentence[i-1].upos in clause_boundary_tags):
                    return True

    return False

def calculate_fronted_adverbial_ratio(doc, pos_tags=None, dep_rels=None, clause_boundary_tags=None):
    """
    Calculate the ratio of sentences with fronted adverbials in a given text,
    with flexibility for POS tags, dependency relations, and clause boundary analysis.
    """
    fronted_adverbials = 0
    total_sentences = len(doc.sentences)

    for sent in doc.sentences:
        # Check the first word and use advanced parsing for better accuracy
        if is_fronted_adverbial(sent.words[0], sent.words, pos_tags, dep_rels, clause_boundary_tags):
            fronted_adverbials += 1
        else:
            # Advanced check for more complex sentence structures
            for word in sent.words[1:5]:  # Focus on the first few words for complex analysis
                if is_fronted_adverbial(word, sent.words, pos_tags, dep_rels, clause_boundary_tags):
                    fronted_adverbials += 1
                    break

    # Normalize the ratio to the range [0, 1]
    ratio = fronted_adverbials / total_sentences if total_sentences > 0 else 0
    return ratio

def is_inverted_structure(sentence):
    subject_positions = []
    verb_positions = []
    expletive_positions = []
    emphatic_positions = []
    adv_positions = []
    potential_subject_positions = []

    ##print(f"\nProcessing sentence: '{sentence.text}'")

    for i, word in enumerate(sentence.words):
        ##print(f"Word: {word.text}, DepRel: {word.deprel}, UPOS: {word.upos}")
        if word.deprel in ('nsubj', 'nsubjpass', 'csubj', 'csubjpass'):
            subject_positions.append(i)
            ##print(f" - Subject found at position {i}")
        elif word.deprel in ('obj', 'iobj') and word.upos == 'NOUN':
            potential_subject_positions.append(i)
            ##print(f" - Potential subject found at position {i}")
        elif word.upos in ('VERB', 'AUX'):
            verb_positions.append(i)
            ##print(f" - Verb found at position {i}")
        elif word.deprel == 'expl':
            expletive_positions.append(i)
            ##print(f" - Expletive found at position {i}")
        elif word.text.lower() in ('here', 'there') and word.deprel == 'advmod':
            emphatic_positions.append(i)
            ##print(f" - Emphatic word found at position {i}")
        elif word.upos == 'ADV':
            adv_positions.append(i)
            ##print(f" - Adverb found at position {i}")

    # Expletive inversion
    if expletive_positions and verb_positions:
        if min(expletive_positions) < min(verb_positions):
            ##print(" -> Expletive inversion detected")
            return "expletive_inversion"

    # Emphatic inversion
    if emphatic_positions and verb_positions:
        if min(emphatic_positions) < min(verb_positions):
            ##print(" -> Emphatic inversion detected")
            return "emphatic_inversion"

    # Classic inversion
    all_subject_positions = subject_positions + potential_subject_positions
    if all_subject_positions and verb_positions:
        if min(all_subject_positions) > min(verb_positions):
            # Check for adverb or prepositional phrase at the beginning
            if (adv_positions and min(adv_positions) == 0) or (sentence.words[0].upos == 'ADP'):
                ##print(" -> Classic inversion detected (with initial adverb or prepositional phrase)")
                return "classic_inversion"
            else:
                ##print(" -> Classic inversion detected")
                return "classic_inversion"

    # If no inversion is found, return None
    ##print(" -> No inversion detected")
    return None

def compute_inversion_frequencies(doc):
    inversion_counts = {
        "classic_inversion": 0,
        "expletive_inversion": 0,
        "emphatic_inversion": 0
    }
    
    total_sentences = len(doc.sentences)

    if total_sentences == 0:
        return {key: 0.0 for key in inversion_counts}

    for sentence in doc.sentences:
        inversion_type = is_inverted_structure(sentence)
        if inversion_type:
            inversion_counts[inversion_type] += 1

    normalized_frequencies = {key: count / total_sentences for key, count in inversion_counts.items()}
    
    return normalized_frequencies

def is_initial_conjunction(token, sentence):
    """Check if a token is a sentence-initial conjunction."""
    # Check if the token is at the start of the sentence
    if token.id == sentence[0].id:
        # Check if the token is a conjunction
        if token.pos == 'CC':
            return True
    return False

def calculate_initial_conjunction_ratio(doc):
    """Calculate the ratio of sentence-initial conjunctions in a given text."""

    initial_conjunctions = 0
    total_sentences = len(doc.sentences)

    for sent in doc.sentences:
        for token in sent.tokens:
            if is_initial_conjunction(token, sent.tokens):
                initial_conjunctions += 1
                break  # Only count one conjunction per sentence

    # Normalize to the range [0, 1]
    ratio = initial_conjunctions / total_sentences if total_sentences > 0 else 0
    return ratio

def is_embedded_clause(token):
    """Check if a token indicates an embedded clause based on its dependency relation."""
    return token.deprel in {"acl", "acl:relcl", "relcl", "ccomp", "xcomp", "advcl"}

def has_embedded_clause(sentence):
    """Check if a sentence has an embedded clause."""
    for word in sentence.words:
        if is_embedded_clause(word):
            ##print(f"DEBUG: Embedded clause detected: {word.text} ({word.deprel})")
            return True
        # Check for reduced relative clauses
        if word.deprel in {"acl", "acl:relcl"} and word.upos == "VERB":
            ##print(f"DEBUG: Reduced relative clause detected: {word.text} ({word.deprel})")
            return True
    return False

def calculate_embedded_clause_ratio(doc):
    """Calculate the ratio of sentences with embedded clauses in a given text."""
    embedded_clauses = 0
    total_sentences = len(doc.sentences)
    
    if total_sentences == 0:
        ##print("DEBUG: No sentences found in the document.")
        return 0.0
    
    for sent in doc.sentences:
        has_embedded = has_embedded_clause(sent)
        ##print(f"DEBUG: Sentence '{sent.text}' has embedded clause: {has_embedded}")
        if has_embedded:
            embedded_clauses += 1
    
    ratio = embedded_clauses / total_sentences
    ##print(f"DEBUG: Total sentences = {total_sentences}, Sentences with embedded clauses = {embedded_clauses}, Ratio = {ratio:.4f}")
    return ratio

def estimated_stressed_syllables(word):
    """
    Estimates the number of stressed syllables in a word using the pronouncing library.
    
    Parameters:
        word (str): The input word.

    Returns:
        int: The number of stressed syllables in the word.
    """
    word = word.lower()
    pronunciations = pronouncing.phones_for_word(word)
    if pronunciations:
        # Get the first pronunciation option (usually the most common)
        pronunciation = pronunciations[0]
        # Extract stress patterns (1 = stressed, 0 = unstressed)
        stress_count = sum(1 for phoneme in pronunciation.split() if phoneme[-1].isdigit() and phoneme[-1] in '12')
        return stress_count
    else:
        return 0

def ratio_of_stressed_syllables(doc):
    """
    Computes the ratio of stressed syllables to total syllables in the given Stanza Document.
    
    Parameters:
        doc (stanza.Document): The Stanza Document object.

    Returns:
        float: The ratio of stressed syllables to total syllables.
    """
    total_syllables = 0
    stressed_syllables = 0

    for sentence in doc.sentences:
        for word in sentence.words:
            word_text = word.text
            syllables = count_syllables(word_text)
            stressed = estimated_stressed_syllables(word_text)
            
            total_syllables += syllables
            stressed_syllables += stressed
    
    return stressed_syllables / total_syllables if total_syllables > 0 else 0.0

def get_stress_pattern(word):
    """
    Get the stress pattern of a word from its phonetic transcription.
    
    Parameters:
        word (str): The input word.
    
    Returns:
        list: A list of stress patterns (1 for stressed, 0 for unstressed).
    """
    word = word.lower()
    stress_patterns = []
    
    pronunciations = pronouncing.phones_for_word(word)
    if pronunciations:
        # Use the first pronunciation option
        pronunciation = pronunciations[0]
        # Extract stress patterns from the pronunciation
        for phone in pronunciation.split():
            # Stress is indicated by digit in the phone (e.g., '1', '2')
            if phone[-1] in '12':  # '1' and '2' indicate stressed syllables
                stress_patterns.append(1)
            else:
                stress_patterns.append(0)
    
    return stress_patterns

def pairwise_variability_index(doc):
    """
    Calculate the Pairwise Variability Index (PVI) for stress patterns.
    
    Parameters:
        stress_patterns (list): A list of stress patterns (1 for stressed, 0 for unstressed).
    
    Returns:
        float: The Pairwise Variability Index (PVI), normalized to the 0-1 range.
    """

    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words if word.text.isalpha()]
    stress_patterns = []
    
    for word in words:
        stress_patterns.extend(get_stress_pattern(word))


    if len(stress_patterns) < 2:
        return 0.0
    
    # Calculate pairwise differences
    differences = [abs(stress_patterns[i] - stress_patterns[i + 1]) for i in range(len(stress_patterns) - 1)]
    
    # Calculate the average difference
    average_difference = np.mean(differences)
    
    # # Normalize the PVI to the range 0-1
    # max_possible_difference = 1
    # normalized_pvi = average_difference / max_possible_difference
    
    return average_difference

def calculate_noun_overlap(doc):
    overlap_count = 0
    previous_nouns = set()

    for sentence in doc.sentences:
        current_nouns = {word.text.lower() for word in sentence.words if word.upos == 'NOUN'}
        if previous_nouns & current_nouns:
            overlap_count += 1
        previous_nouns = previous_nouns | current_nouns  # Keep accumulating nouns

    return overlap_count / len(doc.sentences) if len(doc.sentences) > 0 else 0

def is_rare(word, threshold=0.001):
    """
    Determines if a word is considered rare based on its frequency.

    Parameters:
        word (str): The input word.
        threshold (float): Frequency threshold for rarity.

    Returns:
        bool: True if the word is considered rare, False otherwise.
    """
    # Get the frequency of the word
    freq = word_frequency(word, 'en')
    # Determine if the word's frequency is below the threshold
    return freq < threshold

def rare_words_ratio(doc, threshold=0.001):
    """
    Calculates the ratio of rare words in a given document.

    Parameters:
        doc (stanza.Document): The Stanza Document object.
        threshold (float): Frequency threshold for rarity.

    Returns:
        float: The ratio of rare words normalized to 0-1 range.
    """
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    rare_words = [word for word in words if is_rare(word, threshold)]
    return len(rare_words) / len(words) if words else 0

def zipfian_distribution(x, s, c):
    """Zipf's law with a constant term to handle the long tail."""
    return c / (x ** s)

def compute_zipfian_loss(doc, min_word_length=1, max_rank=None):
    """
    Compute the Zipfian loss for a given document.
    
    :param doc: A stanza Document object
    :param min_word_length: Minimum word length to consider (default: 1)
    :param max_rank: Maximum rank to consider for fitting (default: None, uses all words)
    :return: Zipfian loss value
    """
    # Extract words, filtering by minimum length
    words = [word.text.lower() for sentence in doc.sentences 
             for word in sentence.words 
             if word.text.isalpha() and len(word.text) >= min_word_length]
    
    if len(words) == 0:
        ##print("No valid words found in the document.")
        return np.nan
    
    # Compute frequency distribution
    freq_dist = Counter(words)
    sorted_freqs = sorted(freq_dist.values(), reverse=True)
    
    # Limit ranks if max_rank is specified
    if max_rank is not None and max_rank < len(sorted_freqs):
        sorted_freqs = sorted_freqs[:max_rank]
    
    ranks = np.arange(1, len(sorted_freqs) + 1)
    freqs = np.array(sorted_freqs)
    
    # Fit Zipfian distribution
    try:
        popt, _ = curve_fit(zipfian_distribution, ranks, freqs, p0=[1.0, freqs[0]], 
                            bounds=([0, 0], [np.inf, np.inf]))
        s, c = popt
    except Exception as e:
        #print(f"Error in fitting Zipfian distribution: {e}")
        return np.nan
    
    # Compute fitted frequencies
    fitted_freqs = zipfian_distribution(ranks, s, c)
    
    # Normalize by text length
    total_words = len(words)
    freq_norm = freqs / total_words
    fitted_freq_norm = fitted_freqs / total_words
    
    # Compute loss using Kullback-Leibler divergence
    epsilon = 1e-10  # Small constant to avoid log(0)
    kl_div = np.sum(freq_norm * np.log((freq_norm + epsilon) / (fitted_freq_norm + epsilon)))
    
    # Compute R-squared for goodness of fit
    ss_res = np.sum((freq_norm - fitted_freq_norm) ** 2)
    ss_tot = np.sum((freq_norm - np.mean(freq_norm)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    #print(f"Fitted Zipf's law parameters: s = {s:.4f}, c = {c:.4f}")
    #print(f"R-squared: {r_squared:.4f}")
    #print(f"KL divergence: {kl_div:.6f}")
    
    return kl_div

def average_text_concreteness(text):
    """
    Computes the average concreteness score for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The average concreteness score.
    """
    return avg_text_concreteness(text)

def ratio_concrete_to_abstract(text):
    """
    Computes the ratio of concrete to abstract words in the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The ratio of concrete to abstract words.
    """
    return concrete_abstract_ratio(text)

def pronoun_sentence_opening_ratio(doc):
    """
    Calculates the ratio of sentences in the document that start with a pronoun.

    Parameters:
        doc (stanza.Document): The Stanza processed document.

    Returns:
        float: The ratio of sentences starting with a pronoun.
    """
    total_sentences = len(doc.sentences)
    initial_pronouns = 0

    for sentence in doc.sentences:
        first_word_pos = sentence.words[0].upos
        if first_word_pos == 'PRON':
            initial_pronouns += 1

    # Calculate the ratio of sentences starting with a pronoun
    ratio = initial_pronouns / total_sentences if total_sentences > 0 else 0
    return ratio

def is_sentence_initial_conjunction(sentence):
    """
    Determines if a sentence starts with a conjunction based on POS tags.

    Parameters:
        sentence (stanza.Sentence): A Stanza Sentence object.

    Returns:
        bool: True if the sentence starts with a conjunction, False otherwise.
    """
    # Extract the POS tag of the first word
    if len(sentence.words) > 0:
        first_word_pos = sentence.words[0].pos
        # Conjunction POS tags include CC (Coordinating Conjunction)
        if first_word_pos == 'CC':
            return True
    
    return False

def ratio_of_sentence_initial_conjunctions(doc):
    """
    Computes the ratio of sentences that start with a conjunction in the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The ratio of sentences that start with a conjunction in the text.
    """
    sentences = doc.sentences
    
    if not sentences:
        return 0.0
    
    conjunction_count = sum(1 for sentence in sentences if is_sentence_initial_conjunction(sentence))
    
    ratio = conjunction_count / len(sentences)
    
    return ratio

def summer_index(doc):
    """
    Computes an enhanced Summer's Index for the given text, incorporating syllable counts.
    Parameters:
    doc: A Stanza Document object containing the processed text.
    Returns:
    float: The enhanced Summer's Index value.
    """
    # Extract words from the text, excluding punctuation and whitespace
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words if word.upos != "PUNCT"]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    if total_words == 0:
        return 0
    
    # Calculate the number of unique words (types)
    unique_words = len(set(words))
    
    # Calculate Type-Token Ratio (TTR)
    ttr = unique_words / total_words
    
    # Calculate syllable counts
    total_syllables = 0
    for word in words:
        pronunciations = pronouncing.phones_for_word(word)
        if pronunciations:
            syllable_count = pronouncing.syllable_count(pronunciations[0])
        else:
            # Fallback method if pronunciation is not found
            syllable_count = max(1, len(word) // 3)
        total_syllables += syllable_count
    
    # Calculate average syllables per word
    avg_syllables_per_word = total_syllables / total_words
    
    # Compute enhanced Summer's Index
    base_index = 1 / (1 + math.log10(ttr + 1e-10))  # Base Summer's Index
    syllable_factor = math.log(avg_syllables_per_word + 1, 2)  # Syllable adjustment factor
    enhanced_index = base_index * syllable_factor
    
    return enhanced_index

def is_figurative(sent):
    """
    Heuristic method to detect if a sentence is likely figurative using Stanza.
    
    Parameters:
        sent (stanza.models.common.doc.Sentence): The input Stanza sentence object.
        
    Returns:
        bool: True if the sentence is likely figurative, False otherwise.
    """
    figurative_clues = {"like", "as", "than", "seems", "appears", "metaphorically", "resembles", "is", "are"}

    for word in sent.words:
        if word.text.lower() in figurative_clues:
            if word.deprel in {"case", "mark"} and word.text.lower() in {"like", "as"}:
                return True
            if word.upos == "AUX" and word.text.lower() in {"is", "are"}:
                for child in sent.words:
                    if child.head == word.id and child.upos in {"NOUN", "PRON"}:
                        return True
    return False

def figurative_vs_literal_ratio(doc):
    """
    Calculates the ratio of figurative language versus literal language in the text using Stanza.
    
    Parameters:
        text (str): The input text.
        
    Returns:
        float: The ratio of figurative to literal language.
    """
    figurative_count = sum(is_figurative(sent) for sent in doc.sentences)
    total_sentences = len(doc.sentences)
    
    return figurative_count / total_sentences if total_sentences > 0 else 0.0

def complex_words_rate(doc):
    """
    Computes the rate of complex words (words with three or more syllables) in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The complex words rate.
    """
    # Extract words from the processed document
    words = [word.text for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Calculate the number of complex words (words with three or more syllables)
    complex_words_count = sum(1 for word in words if count_syllables(word) >= 3)
    
    # Calculate the complex words rate
    complex_words_rate = complex_words_count / total_words
    
    return complex_words_rate

def dale_chall_complex_words_rate(doc):
    """
    Computes the Dale-Chall complex words rate for the given text.
    Parameters:
    doc: A Stanza Document object containing the processed text.
    Returns:
    float: The Dale-Chall complex words rate.
    """
    # Extract all words from the document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words if word.upos != "PUNCT"]
    
    # Total number of words
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Reconstruct the text from words
    text = ' '.join(words)
    
    # Calculate the number of complex words using textstat
    dale_chall_complex_words = textstat.difficult_words(text)
    
    # Calculate the complex words rate
    complex_words_rate = dale_chall_complex_words / total_words
    
    return complex_words_rate

def guirauds_index(doc):
    """
    Computes Guiraud's Index for the given text.
    
    Parameters:
        text (str): The input text.
    
    Returns:
        float: The Guiraud's Index value.
    """
    
    # Extract words from the text
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    # Calculate total number of words (tokens)
    total_words = len(words)
    
    if total_words == 0:
        return 0
    
    # Calculate the number of unique words (types)
    unique_words = len(set(words))
    
    # Calculate Guiraud's Index
    guirauds_index_value = unique_words / math.sqrt(total_words)
    
    return guirauds_index_value

def sentence_type_ratio(doc):
    """
    Computes the ratio of different sentence types in the given text.
    
    Parameters:
        doc (stanza.Document): The Stanza Document object containing parsed text.
    
    Returns:
        dict: A dictionary with ratios of different sentence types, normalized to sum to 1.
    """
    
    # Initialize counters for each sentence type with all types preset to zero
    sentence_type_counts = defaultdict(int, {'simple': 0, 'compound': 0, 'complex': 0, 'compound-complex': 0})
    total_sentences = len(doc.sentences)
    
    for sentence in doc.sentences:
        # Count the number of clauses
        num_clauses = sum(1 for word in sentence.words if word.deprel in ['ccomp', 'acl', 'advcl'])
        
        if num_clauses > 1:
            # Compound-Complex
            sentence_type_counts['compound-complex'] += 1
        elif num_clauses == 1:
            # Complex
            sentence_type_counts['complex'] += 1
        elif len(sentence.words) > 1 and any(word.deprel == 'conj' for word in sentence.words):
            # Compound
            sentence_type_counts['compound'] += 1
        else:
            # Simple
            sentence_type_counts['simple'] += 1
    
    # Compute ratios
    if total_sentences > 0:
        sentence_type_ratios = {typ: count / total_sentences for typ, count in sentence_type_counts.items()}
    else:
        # If no sentences, return zero ratios
        sentence_type_ratios = {typ: 0 for typ in ['simple', 'compound', 'complex', 'compound-complex']}
    
    # Ensure the sum of ratios is 1
    total_ratio = sum(sentence_type_ratios.values())
    if total_ratio > 0:
        sentence_type_ratios = {typ: ratio / total_ratio for typ, ratio in sentence_type_ratios.items()}
    
    return sentence_type_ratios
    

def calculate_frazier_depth(sentence):
    """
    Computes the Frazier Depth for a given sentence based on its dependency parse.
    Parameters:
    sentence (stanza.Sentence): A Stanza Sentence object.
    Returns:
    int: The depth of the syntactic structure.
    """
    def build_tree(words):
        tree = {word.id: [] for word in words}
        for word in words:
            if word.head != 0:  # not root
                tree[word.head].append(word.id)
        return tree

    def depth(node_id, tree, current_depth):
        max_depth = current_depth
        for child_id in tree[node_id]:
            max_depth = max(max_depth, depth(child_id, tree, current_depth + 1))
        return max_depth

    # Build the tree
    dependency_tree = build_tree(sentence.words)
    
    # Find the root
    root_id = next(word.id for word in sentence.words if word.head == 0)
    
    # Calculate the depth starting from the root node
    return depth(root_id, dependency_tree, 0)

def frazier_depth(doc):
    """
    Computes the average Frazier Depth for all sentences in a document.
    Parameters:
    doc (stanza.Document): A Stanza Document object.
    Returns:
    float: The average Frazier Depth for the document.
    """
    depths = []
    for sentence in doc.sentences:
        depths.append(calculate_frazier_depth(sentence))
    
    if not depths:
        return 0.0
    
    return sum(depths) / len(depths)

def syll_per_word(doc):
    """
    Computes the average number of syllables per word in the given document using the pronouncing library.

    Parameters:
        doc (stanza.Document): The Stanza Document object.

    Returns:
        float: The average number of syllables per word.
    """
    # Extract words from the Stanza document
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    
    if not words:
        return 0.0

    # Calculate the total number of syllables using pronouncing
    total_syllables = 0
    for word in words:
        syllables = count_syllables(word)
        total_syllables += syllables

    # Calculate the average syllables per word
    syllables_per_word = total_syllables / len(words)
    
    return syllables_per_word

def count_syllables(word):
    """
    Counts the number of syllables in a word using the pronouncing library.

    Parameters:
        word (str): The input word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    pronunciations = pronouncing.phones_for_word(word)
    if pronunciations:
        # Get the first pronunciation option (usually the most common)
        pronunciation = pronunciations[0]
        # Extract syllables from the pronunciation
        syllables = len(pronunciation.split()) - pronunciation.count('0')
        return syllables
    else:
        return 0

def average_sentence_length(doc):
    """
    Computes the average sentence length in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        float: The average sentence length.
    """
    # Extract sentences from the processed document
    sentences = [sentence for sentence in doc.sentences]
    
    # Calculate total number of sentences
    total_sentences = len(sentences)
    
    if total_sentences == 0:
        return 0.0
    
    # Calculate total number of words
    total_words = sum(len(sentence.words) for sentence in sentences)
    
    # Calculate average sentence length
    average_length = total_words / total_sentences
    
    return average_length

def compute_average_dependency_distance(doc):
    """
    Computes the average dependency distance for the given document.
    Parameters:
    doc (stanza.Document): The input Stanza Document.
    Returns:
    float: The average dependency distance.
    """
    distances = []
    for sentence in doc.sentences:
        # Calculate distances for each word in the sentence
        for word in sentence.words:
            if word.head != 0:  # Check if the word has a head (i.e., it's not a root)
                head_word = sentence.words[word.head - 1]  # Get the head word
                distance = abs(word.id - head_word.id)  # Calculate dependency distance
                distances.append(distance)
    
    if not distances:
        # If no distances were calculated, return 0
        return 0
    
    average_distance = np.mean(distances)
    return average_distance

def calculate_cumulative_syntactic_complexity(doc):
    """Calculate the cumulative syntactic complexity of a document."""
    complexity_score = 0
    for sentence in doc.sentences:
        # Increase complexity for each clause, subordination, or coordination
        complexity_score += sum(1 for word in sentence.words if hasattr(word, 'deprel') and word.deprel in {"acl", "relcl", "ccomp", "xcomp", "nsubj"})
    return complexity_score

def average_syntactic_branching_factor(doc):
    """Calculate the syntactic branching factor of a document."""
    total_branches = 0
    for sentence in doc.sentences:
        # Count the number of branches in the parse tree
        branches = sum(1 for token in sentence.words if hasattr(token, 'deprel') and token.deprel in {"acl", "relcl", "ccomp", "xcomp", "nsubj"})
        total_branches += branches
    
    branching_factor = total_branches / len(doc.sentences) if len(doc.sentences) > 0 else 0
    return branching_factor

def calculate_structural_complexity_index(doc):
    """Calculate the structural complexity index of a document."""
    total_sentences = len(doc.sentences)
    total_clauses = 0
    total_length = 0
    
    for sentence in doc.sentences:
        # Count number of clauses and the length of each sentence
        total_clauses += sum(1 for word in sentence.words if hasattr(word, 'deprel') and word.deprel in {"acl", "relcl", "ccomp", "xcomp", "nsubj"})
        total_length += len(sentence.words)
    
    average_sentence_length = total_length / total_sentences if total_sentences > 0 else 0
    average_clauses_per_sentence = total_clauses / total_sentences if total_sentences > 0 else 0
    
    # Combine features into a single index
    # You can customize the formula based on your needs
    structural_complexity_index = (average_clauses_per_sentence + average_sentence_length) / 2
    
    return structural_complexity_index

def lexical_overlap(doc):
    """
    Measures the lexical overlap in the text.

    Parameters:
        doc (stanza.Document): The Stanza Document object.

    Returns:
        float: The lexical overlap ratio.
    """
    words = [word.text.lower() for sentence in doc.sentences for word in sentence.words]
    word_counts = Counter(words)
    overlap = sum(count for count in word_counts.values() if count > 1)
    return overlap / len(words) if words else 0

def compute_yngve_depth(doc):
    """
    Computes the Yngve Depth for the given text.
    
    Parameters:
        text (str): The input text.
    
    Returns:
        float: The Yngve Depth score of the text.
    """
    # Process the text with the Stanza pipeline
    
    yngve_depths = []
    
    for sentence in doc.sentences:
        # Calculate depth for each word in the sentence
        depths = []
        for word in sentence.words:
            depth = 0
            current_word = word
            # Traverse up the tree to find the root and measure depth
            while current_word.head != 0:
                depth += 1
                current_word = sentence.words[current_word.head - 1]
            depths.append(depth)
        
        if depths:
            # Compute Yngve Depth as the sum of inverse depths
            yngve_depth = sum(1 / d for d in depths if d > 0)
            yngve_depths.append(yngve_depth)
    
    if not yngve_depths:
        # If no depths were calculated, return 0
        return 0
    
    # Average Yngve Depth across all sentences
    average_yngve_depth = np.mean(yngve_depths)
    
    return average_yngve_depth

def syntactic_branching_factor(sentence):
    """
    Computes the syntactic branching factor of a sentence.

    Parameters:
        sentence (stanza.Sentence): A Stanza Sentence object.

    Returns:
        float: The average branching factor of the sentence's syntactic parse tree.
    """
    # Create a dictionary to count the number of children for each node
    child_count = {}

    # Iterate through the words to count children
    for word in sentence.words:
        head = word.head
        if head not in child_count:
            child_count[head] = 0
        child_count[head] += 1

    # Compute the average branching factor
    if child_count:
        avg_branching_factor = np.mean(list(child_count.values()))
    else:
        avg_branching_factor = 0.0

    return avg_branching_factor

def branching_factor_for_text(doc):
    """
    Computes the average syntactic branching factor for all sentences in a given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The average syntactic branching factor for the text.
    """
    branching_factors = [syntactic_branching_factor(sentence) for sentence in doc.sentences]
    
    if branching_factors:
        avg_branching_factor = np.mean(branching_factors)
    else:
        avg_branching_factor = 0.0

    return avg_branching_factor

def frequent_delimiters_rate(doc, delimiters={',', '.', '?', '!'}):
    """
    Computes the rate of frequent delimiters in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.
        delimiters (set): A set of frequent delimiter characters to consider.

    Returns:
        float: The rate of frequent delimiters.
    """
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return 0.0
    
    delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
    return delimiter_count / total_words

def lessfrequent_delimiters_rate(doc, delimiters={';', ':'}):
    """
    Computes the rate of less frequent delimiters in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.
        delimiters (set): A set of less frequent delimiter characters to consider.

    Returns:
        float: The rate of less frequent delimiters.
    """
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return 0.0
    
    delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
    return delimiter_count / total_words

def parentheticals_and_brackets_rate(doc, delimiters={'(', ')', '[', ']'}):
    """
    Computes the rate of parentheticals (parentheses and brackets) in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.
        delimiters (set): A set of characters representing parentheses and brackets.

    Returns:
        float: The rate of parentheticals and brackets.
    """
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return 0.0
    
    delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
    return delimiter_count / total_words

# #def quotations_rate(doc, delimiters={'"', "'", '“', '”', '‘', '’', '«', '»', '‹', '›'}):
#     """
#     Computes the rate of quotations in the given processed document.

#     Parameters:
#         doc: The processed document from the Stanza pipeline.
#         delimiters (set): A set of characters representing various types of quotation marks.

#     Returns:
#         float: The rate of quotations.
#     """
#     total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
#     if total_words == 0:
#         return 0.0
    
#     delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
#     return delimiter_count / total_words

def dashes_and_ellipses_rate(doc, delimiters={'—', '–', '‒', '―', '...', '…'}):
    """
    Computes the rate of dashes and ellipses in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.
        delimiters (set): A set of characters representing various types of dashes and ellipses.

    Returns:
        float: The rate of dashes and ellipses.
    """
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return 0.0
    
    delimiter_count = sum(1 for sentence in doc.sentences for token in sentence.tokens if token.text in delimiters)
    
    return delimiter_count / total_words

def compute_hierarchical_structure_complexity(doc):
    """
    Computes the Hierarchical Structure Complexity (HSC) for the given text.
    
    Parameters:
        text (str): The input text.
    
    Returns:
        float: The average hierarchical depth of words in the text.
    """
    
    depths = []
    
    for sentence in doc.sentences:
        # Calculate depth for each word in the sentence
        for word in sentence.words:
            depth = 0
            current_word = word
            # Traverse up the tree to find the root and measure depth
            while current_word.head != 0:
                depth += 1
                current_word = sentence.words[current_word.head - 1]
            depths.append(depth)
    
    if not depths:
        # If no depths were calculated, return 0
        return 0
    
    average_depth = np.mean(depths)
    
    return average_depth

def flesch_reading_ease(text):
    """
    Computes the Flesch Reading Ease score for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The Flesch Reading Ease score.
    """
    return textstat.flesch_reading_ease(text)

def GFI(text):
    """
    Computes the Gunning Fog Index (GFI) for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The Gunning Fog Index.
    """
    return textstat.gunning_fog(text)

def coleman_liau_index(text):
    """
    Computes the Coleman-Liau Index for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The Coleman-Liau Index.
    """
    return textstat.coleman_liau_index(text)

def ari(text):
    """
    Computes the Automated Readability Index (ARI) for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The Automated Readability Index.
    """
    return textstat.automated_readability_index(text)

def dale_chall_readability_score(text):
    """
    Computes the Dale-Chall Readability Score for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The Dale-Chall Readability Score.
    """
    return textstat.dale_chall_readability_score(text)

def lix(text):
    """
    Computes the LIX (Läsbarhetsindex) for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The LIX score.
    """
    return textstat.lix(text)

def smog_index(text):
    """
    Computes the SMOG Index for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The SMOG Index.
    """
    return textstat.smog_index(text)

def rix(text):
    """
    Computes the RIX (RäSvenska LIX) for the given text.

    Parameters:
        text (str): The input text.

    Returns:
        float: The RIX score.
    """
    return textstat.rix(text)

def nominalization(text):
    """
    Computes the normalized nominalization score for the given Stanza document using the readability package.

    Parameters:
        doc (stanza.Document): The input Stanza document.

    Returns:
        float: The normalized nominalization score (nominalization score per word).
    """

    # Get readability measures
    results = getmeasures(text, lang='en')
    
    # Extract the nominalization score
    nominalization_score = results['word usage']['nominalization']
    
    # Calculate the number of words using Stanza's word count
    word_count = len([word for sentence in doc.sentences for word in sentence.words])
    
    # Normalize the nominalization score by dividing by the word count
    normalized_score = nominalization_score / word_count if word_count > 0 else 0
    
    return normalized_score

# #not sure about how applicable this level of textual granularity is....
# def preposition_usage(doc):
#     """
#     Computes the usage rates of specific prepositions in the given processed document.
    
#     Parameters:
#     doc: The processed document from the Stanza pipeline.
    
#     Returns:
#     dict: A dictionary of usage rates for each preposition.
#     """
#     prepositions = ['in', 'of', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after']
#     total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
#     if total_words == 0:
#         return {prep: 0.0 for prep in prepositions}
    
#     preposition_counts = {
#         prep: sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() == prep)
#         for prep in prepositions
#     }
    
#     return {prep: count / total_words for prep, count in preposition_counts.items()}
#this implementation will be based on relative frequencies of prepositions

def preposition_usage(doc):
    """
    Computes the usage rates of specific prepositions in the given processed document,
    standardized against the total number of prepositions.
    
    Parameters:
    doc: The processed document from the Stanza pipeline.
    
    Returns:
    dict: A dictionary of usage rates for each preposition.
    """
    prepositions = ['in', 'of', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after']
    
    # Count the total number of prepositions in the document
    total_prepositions = sum(
        1 for sentence in doc.sentences for word in sentence.words if word.text.lower() in prepositions
    )
    
    if total_prepositions == 0:
        return {prep: 0.0 for prep in prepositions}
    
    # Count the occurrences of each preposition
    preposition_counts = {
        prep: sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() == prep)
        for prep in prepositions
    }
    
    # Standardize against the total number of prepositions
    return {prep: count / total_prepositions for prep, count in preposition_counts.items()}

def detailed_conjunctions_usage(doc):
    """
    Computes the usage rates of different types of conjunctions in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        dict: A dictionary with usage rates for coordinating, subordinating, and correlative conjunctions.
    """
    # Define conjunctions categories
    coordinating_conjunctions = {'and', 'or', 'but', 'so', 'for', 'nor', 'yet'}
    subordinating_conjunctions = {'although', 'because', 'since', 'unless', 'while', 'though', 'if', 'as', 'when', 'after', 'before', 'until', 'where', 'whether'}
    correlative_conjunctions = {'either', 'neither', 'not only', 'both', 'whether'}  # Correlative pairs are handled differently
    
    # Total words in the document
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return {
            "coordinating": 0.0,
            "subordinating": 0.0,
            "correlative": 0.0
        }
    
    # Counts
    coordinating_count = 0
    subordinating_count = 0
    correlative_count = 0

    for sentence in doc.sentences:
        for i, word in enumerate(sentence.words):
            word_lower = word.text.lower()
            if word_lower in coordinating_conjunctions:
                coordinating_count += 1
            elif word_lower in subordinating_conjunctions:
                subordinating_count += 1
            elif word_lower in correlative_conjunctions:
                if word_lower == 'either' and i + 1 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'or':
                    correlative_count += 1
                elif word_lower == 'neither' and i + 1 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'nor':
                    correlative_count += 1
                elif word_lower == 'not' and i + 2 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'only' and sentence.words[i + 2].text.lower() == 'but':
                    correlative_count += 1
                elif word_lower == 'both' and i + 1 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'and':
                    correlative_count += 1
                elif word_lower == 'whether' and i + 1 < len(sentence.words) and sentence.words[i + 1].text.lower() == 'or':
                    correlative_count += 1

    # Calculate the rates
    coordinating_rate = coordinating_count / total_words
    subordinating_rate = subordinating_count / total_words
    correlative_rate = correlative_count / total_words

    return {
        "coordinating": coordinating_rate,
        "subordinating": subordinating_rate,
        "correlative": correlative_rate
    }

def auxiliary_infipart_modals_usage_rate(doc):
    """
    Computes the usage rates of specific auxiliary verbs, infinitives, participles, and modals in the given processed document.

    Parameters:
        doc: The processed document from the Stanza pipeline.

    Returns:
        list: A list of usage rates for each group of words.
    """
    group1_auxiliaries = {'is', 'are', 'was', 'were', 'do', 'does', 'did'}
    group2_auxiliaries = {'have', 'has', 'had'}
    infinitives_participles = {'be', 'being', 'been'}
    modals = {'can', 'could', 'should', 'will', 'would', 'may', 'might', 'must'}
    
    total_words = sum(len(sentence.words) for sentence in doc.sentences)
    
    if total_words == 0:
        return [0.0] * 4
    
    group1_count = sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() in group1_auxiliaries)
    group2_count = sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() in group2_auxiliaries)
    infinitives_participles_count = sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() in infinitives_participles)
    modals_count = sum(1 for sentence in doc.sentences for word in sentence.words if word.text.lower() in modals)
    
    return [
        group1_count / total_words,
        group2_count / total_words,
        infinitives_participles_count / total_words,
        modals_count / total_words
    ]
