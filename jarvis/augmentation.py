import random
import json
import nlpaug.augmenter.word as naw
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import spacy

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class DataAugmenter:
    def __init__(self):
        self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
        self.back_translation_aug = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en'
        )
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            import os
            os.system('python -m spacy download en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
    
    def synonym_replacement(self, text: str, n_words: int = 1) -> str:
        """Replace n words with their synonyms."""
        words = word_tokenize(text)
        n = min(n_words, len(words))
        positions = random.sample(range(len(words)), n)
        
        for pos in positions:
            synonyms = []
            for syn in wordnet.synsets(words[pos]):
                for lemma in syn.lemmas():
                    if lemma.name() != words[pos]:
                        synonyms.append(lemma.name())
            
            if synonyms:
                words[pos] = random.choice(synonyms)
        
        return ' '.join(words)
    
    def back_translation(self, text: str) -> str:
        """Augment text using back translation."""
        try:
            augmented_text = self.back_translation_aug.augment(text)[0]
            return augmented_text
        except:
            return text
    
    def entity_replacement(self, text: str) -> str:
        """Replace named entities with similar entities."""
        doc = self.nlp(text)
        words = list(text.split())
        
        entity_replacements = {
            'PERSON': ['John', 'Alice', 'Bob', 'Emma', 'David'],
            'ORG': ['Google', 'Microsoft', 'Apple', 'Amazon', 'Facebook'],
            'GPE': ['London', 'Paris', 'Tokyo', 'New York', 'Berlin'],
            'DATE': ['tomorrow', 'next week', 'next month', 'today', 'yesterday'],
            'TIME': ['9 AM', '3 PM', '6:30 PM', 'noon', 'midnight']
        }
        
        for ent in doc.ents:
            if ent.label_ in entity_replacements:
                replacement = random.choice(entity_replacements[ent.label_])
                text = text.replace(ent.text, replacement)
        
        return text
    
    def random_insertion(self, text: str, n_words: int = 1) -> str:
        """Insert n random words from wordnet into the text."""
        words = text.split()
        for _ in range(n_words):
            add_word = random.choice(list(wordnet.words()))
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, add_word)
        
        return ' '.join(words)
    
    def augment_data(
        self,
        data: List[Dict],
        augmentation_factor: int = 2,
        techniques: List[str] = None
    ) -> List[Dict]:
        """
        Augment the dataset using multiple techniques.
        
        Args:
            data: List of dictionaries containing text and label
            augmentation_factor: How many times to augment each example
            techniques: List of techniques to use ['synonym', 'backtranslation', 'entity', 'insertion']
        """
        if techniques is None:
            techniques = ['synonym', 'backtranslation', 'entity', 'insertion']
            
        augmented_data = []
        
        for item in data:
            text = item['text']
            label = item['label']
            
            # Add original data
            augmented_data.append(item)
            
            # Add augmented versions
            for _ in range(augmentation_factor - 1):
                technique = random.choice(techniques)
                augmented_text = text
                
                if technique == 'synonym':
                    augmented_text = self.synonym_replacement(text)
                elif technique == 'backtranslation':
                    augmented_text = self.back_translation(text)
                elif technique == 'entity':
                    augmented_text = self.entity_replacement(text)
                elif technique == 'insertion':
                    augmented_text = self.random_insertion(text)
                
                augmented_data.append({
                    'text': augmented_text,
                    'label': label
                })
        
        return augmented_data

# Example usage:
"""
# Load original data
with open('data/train.json', 'r') as f:
    data = json.load(f)

# Initialize augmenter
augmenter = DataAugmenter()

# Augment data with all techniques
augmented_data = augmenter.augment_data(
    data,
    augmentation_factor=3,
    techniques=['synonym', 'backtranslation', 'entity', 'insertion']
)

# Save augmented data
with open('data/augmented_train.json', 'w') as f:
    json.dump(augmented_data, f, indent=2)

# Example of individual augmentations
text = "Open Chrome and play some music"
print("Original:", text)
print("Synonym:", augmenter.synonym_replacement(text))
print("Back Translation:", augmenter.back_translation(text))
print("Entity:", augmenter.entity_replacement(text))
print("Insertion:", augmenter.random_insertion(text))
"""
