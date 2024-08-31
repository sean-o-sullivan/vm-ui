import pandas as pd
import numpy as np
import os
import ast



print('hi')
def create_feature_mapping():
    # Create a mapping of index positions to feature names
    feature_names = [
        'compute_herdans_v', 'compute_brunets_w', 'tuldava_ln_ttr', 'simpsons_index', 'sichel_s_measure',
        'orlov_sigma', 'pos_frequencies_pos_frequencies_ADJ', 'pos_frequencies_pos_frequencies_ADP',
        'pos_frequencies_pos_frequencies_ADV', 'pos_frequencies_pos_frequencies_AUX',
        'pos_frequencies_pos_frequencies_CCONJ', 'pos_frequencies_pos_frequencies_DET',
        'pos_frequencies_pos_frequencies_INTJ', 'pos_frequencies_pos_frequencies_NOUN',
        'pos_frequencies_pos_frequencies_NUM', 'pos_frequencies_pos_frequencies_PART',
        'pos_frequencies_pos_frequencies_PRON', 'pos_frequencies_pos_frequencies_PROPN',
        'pos_frequencies_pos_frequencies_PUNCT', 'pos_frequencies_pos_frequencies_SCONJ',
        'pos_frequencies_pos_frequencies_SYM', 'pos_frequencies_pos_frequencies_VERB',
        'pos_frequencies_pos_frequencies_X', 'yules_k_characteristic', 'honores_r', 'renyis_entropy',
        'perplexity', 'burstiness', 'hapax_dislegomena_rate', 'dugast_vmi', 'average_word_length',
        'prosodic_patterns', 'compute_inversion_frequencies_classic_inversion',
        'compute_inversion_frequencies_expletive_inversion',
        'compute_inversion_frequencies_emphatic_inversion', 'clauses_per_sentence',
        'modifiers_per_noun_phrase', 'coordinated_phrases_per_sentence', 'coordinate_phrases_ratio',
        'sentence_length_variation', 'clause_length_variation', 'dependent_clauses_ratio',
        'subordination_index', 'average_sentence_depth', 'compute_average_dependency_distance',
        'frazier_depth', 'branching_factor_for_text', 'compute_hierarchical_structure_complexity',
        'compute_yngve_depth', 'long_word_rate', 'short_word_rate', 'lexical_density',
        'calculate_embedded_clause_ratio', 'ratio_of_stressed_syllables', 'pairwise_variability_index',
        'compute_zipfian_loss', 'frequent_delimiters_rate', 'preposition_usage_in', 'preposition_usage_of',
        'preposition_usage_to', 'preposition_usage_for', 'preposition_usage_with', 'preposition_usage_on',
        'preposition_usage_at', 'preposition_usage_by', 'preposition_usage_from', 'preposition_usage_up',
        'preposition_usage_about', 'preposition_usage_into', 'preposition_usage_over',
        'preposition_usage_after', 'lessfrequent_delimiters_rate', 'parentheticals_and_brackets_rate',
        'calculate_cumulative_syntactic_complexity', 'average_syntactic_branching_factor',
        'calculate_structural_complexity_index', 'lexical_overlap',
        'analyze_passiveness_average_passiveness', 'analyze_passiveness_std_passiveness',
        'ratio_of_cleft_sentences', 'calculate_noun_overlap', 'pronoun_sentence_opening_ratio',
        'ratio_of_sentence_initial_conjunctions', 'calculate_fronted_adverbial_ratio',
        'rare_words_ratio', 'summer_index', 'dale_chall_complex_words_rate', 'guirauds_index',
        'syll_per_word', 'average_sentence_length', 'normalized_assonance', 'normalized_alliteration',
        'tempo_variation', 'rhythmic_complexity', 'complex_words_rate',
        'detailed_conjunctions_usage_coordinating', 'detailed_conjunctions_usage_subordinating',
        'detailed_conjunctions_usage_correlative', 'auxiliary_infipart_modals_usage_rate_1',
        'auxiliary_infipart_modals_usage_rate_2', 'auxiliary_infipart_modals_usage_rate_3',
        'auxiliary_infipart_modals_usage_rate_4', 'sentence_type_ratio_simple',
        'sentence_type_ratio_compound', 'sentence_type_ratio_complex',
        'sentence_type_ratio_compound-complex', 'figurative_vs_literal_ratio', 'flesch_reading_ease',
        'GFI', 'coleman_liau_index', 'ari', 'dale_chall_readability_score', 'lix', 'smog_index', 'rix'
    ]
    return {i: name for i, name in enumerate(feature_names)}

def normalize_future_csv(csv_path, stats_data_path, features_to_omit, output_dir):
    # Load the statistics
    stats_data = pd.read_csv(stats_data_path, index_col=0)
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Determine which column contains the embedding based on the CSV structure
    if 'generated_mimicry_embedding' in df.columns:
        embedding_column = 'generated_mimicry_embedding'
    elif 'generated_text_embedding' in df.columns:
        embedding_column = 'generated_text_embedding'
    else:
        raise ValueError("No recognized embedding column found in the CSV.")

    # Create feature mapping
    feature_mapping = create_feature_mapping()

    # Function to normalize a single embedding
    def normalize_embedding(embedding_str):
        try:
            embedding = ast.literal_eval(embedding_str)
        except:
            print(f"Error parsing embedding: {embedding_str}")
            return embedding_str  # Return original string if parsing fails
        
        normalized_embedding = []
        
        for i, value in enumerate(embedding):
            if i in feature_mapping:
                feature_name = feature_mapping[i]
                if feature_name not in features_to_omit and feature_name in stats_data.index:
                    mean = stats_data.at[feature_name, 'mean']
                    std = stats_data.at[feature_name, 'std']
                    percentile_99_5 = stats_data.at[feature_name, 'percentile_99.5']
                    percentile_0_5 = stats_data.at[feature_name, 'percentile_0.5']
                    
                    if std != 0:
                        z_score = (value - mean) / std
                        z_score_0_5 = (percentile_0_5 - mean) / std
                        z_score_99_5 = (percentile_99_5 - mean) / std
                        normalized_value = (z_score - z_score_0_5) / (z_score_99_5 - z_score_0_5)
                        normalized_value = np.clip(normalized_value, 0, 1)
                    else:
                        normalized_value = 0.5
                    
                    normalized_embedding.append(normalized_value)
        
        return str(normalized_embedding)  # Convert back to string for CSV storage

    # Normalize only the embedding column
    df[embedding_column] = df[embedding_column].apply(normalize_embedding)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the CSV with only the embedding column changed
    output_file_path = os.path.join(output_dir, f"normalized_{os.path.basename(csv_path)}")
    df.to_csv(output_file_path, index=False)
    
    print(f"Normalized CSV saved to: {output_file_path}")

# Main execution
stats_data_path = 'embedding_stats.csv'
features_to_omit = [
    "ratio_of_sentence_initial_conjunctions",
    "detailed_conjunctions_usage_correlative",
    "normalized_assonance"
]
output_dir = 'normalized_adversarial_csvs'

# List of all adversarial testing CSVs with embeddings
# adversarial_csvs = [
#     '../mimicry_samples_GPT3ABB_30_embeddings.csv',
#     '../mimicry_samples_GPT3AGG_30_embeddings.csv',
#     '../mimicry_samples_GPT4oABB_30_embeddings.csv',
#     '../mimicry_samples_GPT4oAGG_30_embeddings.csv',
#     '../mimicry_samples_GPT4TABB_30_embeddings.csv',
#     '../mimicry_samples_GPT4TAGG_30_embeddings.csv',
#     '../topic_based_samples_GPT3ABB_30_embeddings.csv',
#     '../topic_based_samples_GPT3AGG_30_embeddings.csv',
#     '../topic_based_samples_GPT4oABB_30_embeddings.csv',
#     '../topic_based_samples_GPT4oAGG_30_embeddings.csv',
#     '../topic_based_samples_GPT4TABB_30_embeddings.csv',
#     '../topic_based_samples_GPT4TAGG_30_embeddings.csv'
# ]

adversarial_csvs = [
    '/home/aiadmin/Downloads/output_2embeddings_Reuters.csv'
]

# Process all adversarial CSVs
for csv_file in adversarial_csvs:
    normalize_future_csv(csv_file, stats_data_path, features_to_omit, output_dir)
