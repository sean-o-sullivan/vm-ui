import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import spacy
import nltk
from nltk.corpus import words


nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')
nltk.download('punkt')


nlp = spacy.load('en_core_web_lg')


english_words = set(words.words())


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = 'gpt2-large'

def load_transformer_model_and_tokenizer():
    try:
        print("Installing/Locading models")
        model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        print("Model and Tokenizer loaded successfully.")
    except EnvironmentError as e:
        print(f"Failed to load Model and Tokenizer. Error: {str(e)}")
        model = None
        tokenizer = None
    return model, tokenizer

model, tokenizer = load_transformer_model_and_tokenizer()
