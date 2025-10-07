#!/usr/bin/env python3
import pandas as pd  # CSV dosyasını okumak ve veri analizi için
import spacy  # NLP işlemleri için güçlü bir kütüphane
from collections import Counter  # Frekans sayımı için

nlp = spacy.load('en_core_web_sm')

def extract_opinion_phrases(text):
    """
    ADJ: Adjective (sıfat) - good, bad, excellent
    NOUN: İsim - phone, battery, service
    PROPN: Özel isim - iPhone, Samsung
    ADV: Adverb (zarf) - very, really, quite
    """
    doc = nlp(text)  # Metni spaCy ile işle (tokenize + POS tag)
    opinions = []
    
    # Her kelimeyi (token) sırayla incele
    for i, token in enumerate(doc):
        # Sıfat + İsim
        if token.pos_ == 'ADJ' and i + 1 < len(doc) and doc[i + 1].pos_ in ['NOUN', 'PROPN']:
            opinions.append(f"{token.text} {doc[i + 1].text}".lower())
        
        # Zarf + Sıfat
        if token.pos_ == 'ADV' and i + 1 < len(doc) and doc[i + 1].pos_ == 'ADJ':
            opinions.append(f"{token.text} {doc[i + 1].text}".lower())
    
    return opinions

def extract_entities(text):
    """
    ORG: Organization (kuruluş/marka) - Apple, Samsung, Google
    PRODUCT: Ürün adları - iPhone, Galaxy
    GPE: Geopolitical Entity (ülke/şehir) - Turkey, Istanbul
    LOC: Location (konum) - Amazon Store, Mall"""

    doc = nlp(text)
    entities = {'products': [], 'brands': [], 'locations': []}
    
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT']:
            entities['brands'].append(ent.text)
        elif ent.label_ in ['GPE', 'LOC']:
            entities['locations'].append(ent.text)
    
    return entities

def extract_aspect_opinion_pairs(text):
    doc = nlp(text)
    pairs = []
    
    for chunk in doc.noun_chunks:
        aspect = chunk.text.lower().strip()

        for i in range(max(0, chunk.start - 2), min(len(doc), chunk.end + 2)):
            if doc[i].pos_ == 'ADJ' and not doc[i].is_stop:
                pairs.append((aspect, doc[i].text.lower()))
    
    return pairs

df = pd.read_csv('/Users/murat/Desktop/NLP/ikinci_odev/amazon.csv')

print("Amazon Product Reviews NLP Analysis")
print(f"Total reviews: {len(df)}")

all_opinions = []
all_entities = {'products': [], 'brands': [], 'locations': []}
all_pairs = []

for text in df['Text']:
    all_opinions.extend(extract_opinion_phrases(text))
    
    entities = extract_entities(text)
    for key in all_entities:
        all_entities[key].extend(entities[key])
    
    all_pairs.extend(extract_aspect_opinion_pairs(text))

# EN YAYGIN GÖRÜŞ İFADELERİ
print("(Müşterilerin en sık kullandığı görüş ifadeleri)")
for phrase, count in Counter(all_opinions).most_common(10):
    print(f"  {phrase}: {count} kez")

# TESPİT EDİLEN MARKALAR VE YERLER
print("(Yorumlarda bahsedilen marka ve yer adları)")
for entity_type, entities in all_entities.items():
    if entities:  # Eğer bu kategoride entity bulunduysa
        print(f"  {entity_type.title()}:")
        for entity, count in Counter(entities).most_common(5):
            print(f"    {entity}: {count} kez")

# EN YAYGIN ASPECT-OPINION ÇİFTLERİ  
print("(Hangi özellikler hakkında hangi görüşler belirtiliyor?)")
for (aspect, opinion), count in Counter(all_pairs).most_common(10):
    print(f"  {aspect} -> {opinion}: {count} kez")