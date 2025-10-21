import re
from collections import Counter
import pandas as pd
import spacy

CSV_PATH = "/Users/murat/Desktop/NLP/ikinci_odev/amazon.csv"
TOP_OPINIONS = 10
TOP_ENTITIES = 5
TOP_PAIRS = 10

def build_nlp():
    # varsa md modeli, yoksa sm
    try:
        nlp = spacy.load("en_core_web_md")
    except Exception:
        nlp = spacy.load("en_core_web_sm")  # default fallback

    # EntityRuler: Domain'e özel düzeltmeler.
    # Amaç: 'Kindle' ve 'Kindle Fire' gibi cihaz/ürünlerin LOC/GPE'ye kaymasını engellemek,
    # 'Angry Birds' ve 'Amazon Silk' gibi ürünleri PRODUCT'a çekmek.
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns([
        {"label": "PRODUCT", "pattern": "Kindle"},
        {"label": "PRODUCT", "pattern": [{"LOWER":"kindle"},{"LOWER":"fire"}]},
        {"label": "PRODUCT", "pattern": "Angry Birds"},
        {"label": "PRODUCT", "pattern": [{"LOWER":"amazon"},{"LOWER":"silk"}]},
        {"label": "ORG", "pattern": "Amazon"},
        {"label": "ORG", "pattern": "Google"},
        {"label": "ORG", "pattern": "Android"},
        {"label": "ORG", "pattern": "Pandora"},
    ])
    return nlp

nlp = build_nlp()

# Post-processing kuralları:
# - ENTITY_STOPWORDS: 'free', 'faotd' gibi büyük harfli gürültüyü (yanlış ORG/PRODUCT) elemek
# - UPPER_EXCEPTIONS: 'US', 'UK' gibi gerçekten yer olan kısaltmaları korumak
# - BRAND_WHITELIST / PRODUCT_WHITELIST: domain bilgisi ile doğru sınıfa zorlamak
ENTITY_STOPWORDS = {"free", "faotd"}
UPPER_EXCEPTIONS = {"US", "UK", "USA", "EU"}
BRAND_WHITELIST = {"amazon", "google", "android", "pandora"}
PRODUCT_WHITELIST = {"kindle", "kindle fire", "angry birds", "amazon silk"}

_det_pat = re.compile(r"^(the|a|an)\s+", flags=re.I)

def normalize_ent(text: str) -> str:
    # Öğrenci yorumu: 'the Kindle Fire' -> 'Kindle Fire' gibi determinersız forma getiriyorum.
    t = text.strip()
    t = _det_pat.sub("", t)
    return t

def looks_fake_allcaps(text: str) -> bool:
    # 'FREE', 'FAOTD' gibi tamamen büyük harfli ama yer adı olmayan gürültüyü ele.
    return text.isupper() and text not in UPPER_EXCEPTIONS and len(text) >= 3

def is_location_context(doc: spacy.tokens.Doc, ent: spacy.tokens.Span) -> bool:
    # Lokasyon kabulü için minimal bağlam denetimi: 'in/at/from/to X'
    # 'on Kindle' cihaz gibi durumları lokasyon saymıyoruz.
    prev_tok = doc[ent.start - 1] if ent.start > 0 else None
    return bool(prev_tok and prev_tok.lemma_ in {"in", "at", "from", "to"})

# --- Çıkarımlar ---
def extract_opinion_phrases_from_doc(doc: spacy.tokens.Doc):
    # ADJ+NOUN/PROPN ve ADV+ADJ kalıpları ile basit opinion phrase çıkarımı
    opinions = []
    for i, tok in enumerate(doc):
        if tok.pos_ == "ADJ" and i + 1 < len(doc) and doc[i + 1].pos_ in {"NOUN", "PROPN"}:
            opinions.append(f"{tok.text} {doc[i + 1].text}".lower())
        if tok.pos_ == "ADV" and i + 1 < len(doc) and doc[i + 1].pos_ == "ADJ":
            opinions.append(f"{tok.text} {doc[i + 1].text}".lower())
    return opinions

def extract_aspect_opinion_pairs_from_doc(doc: spacy.tokens.Doc):
    # noun_chunk (aspect) çevresinde +/-2 token yakınında ADJ arıyorum
    pairs = []
    for chunk in doc.noun_chunks:
        aspect = chunk.text.lower().strip()
        start = max(0, chunk.start - 2)
        end = min(len(doc), chunk.end + 2)
        for i in range(start, end):
            if doc[i].pos_ == "ADJ" and not doc[i].is_stop:
                pairs.append((aspect, doc[i].text.lower()))
    return pairs

def extract_entities_from_doc(doc: spacy.tokens.Doc):
    # NER sonrası domain odaklı filtreleme. Hedef: false pozitifleri azaltmak.
    entities = {"products": [], "brands": [], "locations": []}
    for ent in doc.ents:
        raw = normalize_ent(ent.text)
        low = raw.lower()

        # Gürültü temizliği
        if low in ENTITY_STOPWORDS:
            continue
        if looks_fake_allcaps(raw):
            continue

        # Domain whitelist
        if low in PRODUCT_WHITELIST:
            entities["products"].append(raw); continue
        if low in BRAND_WHITELIST:
            entities["brands"].append(raw); continue

        if ent.label_ == "PRODUCT":
            entities["products"].append(raw)
        elif ent.label_ == "ORG":
            entities["brands"].append(raw)
        elif ent.label_ in {"GPE", "LOC"}:
            if is_location_context(doc, ent) or raw in UPPER_EXCEPTIONS:
                entities["locations"].append(raw)
    return entities

def main():
    # CSV dosyasını doğrudan oku (kolon adı kesin "Text")
    df = pd.read_csv(CSV_PATH)
    texts = df["Text"].dropna().astype(str).tolist()

    print("Amazon Product Reviews NLP Analysis")
    print(f"Total reviews: {len(texts)}\n")

    all_opinions = []
    all_entities = {"products": [], "brands": [], "locations": []}
    all_pairs = []

    for doc in nlp.pipe(texts, batch_size=64):
        all_opinions.extend(extract_opinion_phrases_from_doc(doc))
        ents = extract_entities_from_doc(doc)
        for k in all_entities:
            all_entities[k].extend(ents[k])
        all_pairs.extend(extract_aspect_opinion_pairs_from_doc(doc))

    print("(Müşterilerin en sık kullandığı görüş ifadeleri)")
    for phrase, count in Counter(all_opinions).most_common(TOP_OPINIONS):
        print(f"  {phrase}: {count} kez")

    print("(Yorumlarda bahsedilen marka, ürün ve yer adları)")
    for entity_type in ["brands", "products", "locations"]:
        entities = all_entities[entity_type]
        if entities:
            print(f"  {entity_type.title()}:")
            for entity, count in Counter(entities).most_common(TOP_ENTITIES):
                print(f"    {entity}: {count} kez")

    print("(Hangi özellikler hakkında hangi görüşler belirtiliyor?)")
    for (aspect, opinion), count in Counter(all_pairs).most_common(TOP_PAIRS):
        print(f"  {aspect} -> {opinion}: {count} kez")

if __name__ == "__main__":
    main()
