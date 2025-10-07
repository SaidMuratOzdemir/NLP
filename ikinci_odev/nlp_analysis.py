#!/usr/bin/env python3
import pandas as pd
import spacy
from collections import Counter

nlp = spacy.load('en_core_web_sm')

def extract_opinion_phrases(text):
    """POS tagging ile opinion phrase'leri tespit et"""
    doc = nlp(text)
    opinions = []
    
    for i, token in enumerate(doc):
        # Adjective + Noun kombinasyonları
        if token.pos_ == 'ADJ' and i + 1 < len(doc) and doc[i + 1].pos_ in ['NOUN', 'PROPN']:
            opinions.append(f"{token.text} {doc[i + 1].text}".lower())
        
        # Adverb + Adjective kombinasyonları
        if token.pos_ == 'ADV' and i + 1 < len(doc) and doc[i + 1].pos_ == 'ADJ':
            opinions.append(f"{token.text} {doc[i + 1].text}".lower())
    
    return opinions

def extract_entities(text):
    """NER ile entity'leri tespit et"""
    doc = nlp(text)
    entities = {'products': [], 'brands': [], 'locations': []}
    
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT']:
            entities['brands'].append(ent.text)
        elif ent.label_ in ['GPE', 'LOC']:
            entities['locations'].append(ent.text)
    
    return entities

def extract_aspect_opinion_pairs(text):
    """POS + Chunking ile aspect-opinion çiftlerini tespit et"""
    doc = nlp(text)
    pairs = []
    
    for chunk in doc.noun_chunks:
        aspect = chunk.text.lower().strip()
        
        # Yakındaki sıfatları bul
        for i in range(max(0, chunk.start - 2), min(len(doc), chunk.end + 2)):
            if doc[i].pos_ == 'ADJ' and not doc[i].is_stop:
                pairs.append((aspect, doc[i].text.lower()))
    
    return pairs

# Ana analiz
df = pd.read_csv('/Users/murat/Desktop/NLP/ikinci_odev/amazon.csv')

print("Amazon Product Reviews NLP Analysis")
print(f"Total reviews: {len(df)}")

all_opinions = []
all_entities = {'products': [], 'brands': [], 'locations': []}
all_pairs = []

# Tüm metinleri analiz et
for text in df['Text']:
    all_opinions.extend(extract_opinion_phrases(text))
    
    entities = extract_entities(text)
    for key in all_entities:
        all_entities[key].extend(entities[key])
    
    all_pairs.extend(extract_aspect_opinion_pairs(text))

# Sonuçları göster
print("\nTop Opinion Phrases:")
for phrase, count in Counter(all_opinions).most_common(10):
    print(f"  {phrase}: {count}")

print("\nDetected Entities:")
for entity_type, entities in all_entities.items():
    if entities:
        print(f"  {entity_type.title()}:")
        for entity, count in Counter(entities).most_common(5):
            print(f"    {entity}: {count}")

print("\nAspect-Opinion Pairs:")
for (aspect, opinion), count in Counter(all_pairs).most_common(10):
    print(f"  {aspect} -> {opinion}: {count}")