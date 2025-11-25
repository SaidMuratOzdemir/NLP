# Lab 3 Not Defteri Satır Satır Açıklama

Aşağıdaki notlar, `Lab_3-TextClass-Murat.ipynb` defterindeki her bir hücreyi ve kod satırını açıklamak için hazırlandı. Böylece hem modeli nasıl kurduğumu hatırlıyorum hem de hocamla konuşurken her adımın nedenini anlatabilecek durumda oluyorum. Not defterine dokunmadan ayrı bir dosyada tüm akışı belgeledim.

---

## Hücre 0 – Başlık (Markdown)
- `# Text Classification with Scikit-Learn` → Çalışmanın ana başlığını veriyor.
- Boş satır ve ikinci paragraf → Laboratuvarın genel amacını bir cümleyle özetliyor.

## Hücre 1 – Amaçlar (Markdown)
- `## Objectives` başlığı → Bölüm başlığı.
- Her madde → Ödevde tamamlanması gereken işleri listeliyor (veri yükleme, ön işleme, modelleme, değerlendirme ve deneyler).

## Hücre 2 – Kurulum Açıklaması (Markdown)
- `## Setup` → Kurulum bölümünün başlığı.
- Metin → Bir sonraki kod hücresindeki importların neden gerekli olduğunu anlatıyor.

## Hücre 3 – Kütüphanelerin ve SSL yapılandırmasının kurulumu
- `import numpy as np` → Sayısal işlemler için NumPy kısayolu.
- `import pandas as pd` → Veri manipülasyonu için pandas.
- `import matplotlib.pyplot as plt` → Çizim fonksiyonları.
- `import seaborn as sns` → Matplotlib üzerine kurulu görselleştirme kütüphanesi.
- `import certifi` → Sertifika deposu sağlayarak HTTPS indirmelerinde SSL hatalarını gidermek için.
- `import ssl` → SSL/TLS yapılandırmalarını ayarlamak için standart kütüphane.
- `import urllib.request` → scikit-learn veri seti indirirken SSL bağlamını kullanmak üzere HTTP istemcisi.
- `from sklearn.datasets import fetch_20newsgroups` → 20 Newsgroups veri setini çeken yardımcı fonksiyon.
- `from sklearn.model_selection import train_test_split, GridSearchCV` → Veri setini ayırmak ve hiperparametre araması yapmak için.
- `from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer` → Metni sayısal özelliğe çeviren vektörleştiriciler.
- `from sklearn.naive_bayes import MultinomialNB` → Naive Bayes sınıflandırıcısı.
- `from sklearn.linear_model import LogisticRegression` → Lojistik regresyon sınıflandırıcısı.
- `from sklearn.svm import LinearSVC` → Lineer destek vektör makinesi sınıflandırıcısı.
- `from sklearn.metrics import (...)` bloğu → Doğruluk, precision, recall, F1, rapor ve karışıklık matrisi çizimi için metrik fonksiyonları içe aktarılıyor.
- `from sklearn.decomposition import TruncatedSVD` → Boyut azaltma için TruncatedSVD sınıfı.
- `from sklearn.pipeline import Pipeline` → Adımları tek bir pipeline içinde birleştirmek için.
- `from sklearn.base import clone` → Estimatörleri orijinalini bozmadan kopyalamak (her kombinasyona sıfırdan başlamak) için.
- `import nltk` → Natural Language Toolkit ana paketi.
- `from nltk.corpus import stopwords` → İngilizce stopword listesini kullanmak için.
- `from nltk.stem import WordNetLemmatizer, PorterStemmer` → Lemmatizasyon ve stemming sınıfları.
- `import re` → Regex ile noktalama işaretlerini temizlemek için.
- `np.random.seed(42)` → NumPy rastgele işlemlerinde reproducible sonuç sağlamak.
- `sns.set_theme(style="whitegrid")` → Seaborn grafikleri için beyaz ızgaralı tema.
- `ssl_context = ssl.create_default_context(cafile=certifi.where())` → Certifi deposunu kullanarak güvenilir SSL bağlamı oluşturur.
- `urllib.request.install_opener(...)` → Oluşturulan SSL bağlamı ile çalışan bir opener kurarak 20 Newsgroups indiriminin sorunsuz olması sağlanır.

## Hücre 4 – NLTK kaynakları indirme
- `downloads = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']` → Gerekli tüm NLTK paketlerini bir listede toplar (tokenizer, stopword, WordNet sözlüğü, lemmatization için ek veri).
- `for resource in downloads:` → Listedeki her kaynak üzerinde döngü.
- `    nltk.download(resource, quiet=True)` → Her kaynağı sessiz modda indirir; notebook çıktısı temiz kalır.

## Hücre 5 – Veri Yükleme Bölüm Başlığı (Markdown)
- Başlık ve açıklama → 20 Newsgroups veri setinin içeriğini ve neden belirli parçaların çıkarıldığını (header/footer/quotes) hatırlatıyor.

## Hücre 6 – Veri setini belleğe alma
- `newsgroups = fetch_20newsgroups(...)` → Tüm veriyi çeker, eşitlik sağlamak için başlık/dipnot/alıntıları çıkarır.
- `texts = newsgroups.data` → Haber metinlerini liste halinde değişkene kaydeder.
- `targets = newsgroups.target` → Her metnin sınıf indekslerini alır.
- `target_names = newsgroups.target_names` → Sınıf isimleri listesini saklar.
- `print(f'Total documents: {len(texts)} across {len(target_names)} topics.')` → Veri setinin boyutunu kullanıcıya bildirir.

## Hücre 7 – Sınıf dağılımını inceleme
- `label_series = pd.Series([target_names[idx] for idx in targets])` → Her hedef indeksini gerçek sınıf adına dönüştürüp pandas Series oluşturur.
- `label_counts = label_series.value_counts().sort_values(ascending=False)` → Sınıfları frekansına göre azalan sıralar; dengesizlik var mı incelenir.
- `label_counts.head(10)` → İlk 10 sınıfın sayısını görüntüler (DataFrame çıktısı).

## Hücre 8 – Rastgele örnekler yazdırma
- `sample_indices = np.random.choice(len(texts), size=3, replace=False)` → Tekrarsız üç rastgele belge seçer.
- `for idx in sample_indices:` → Seçilen indeksler üzerinde döngü.
- `    print('-' * 80)` → Ayraç çizgisi basarak çıktıyı okunaklı kılar.
- `    print(f'Label: {target_names[targets[idx]]}')` → Belgenin gerçek sınıf adını yazar.
- `    print(texts[idx][:500].strip())` → Metnin ilk 500 karakterini kırpıp yazdırır; veri kalitesi görülür.

## Hücre 9 – Ön işleme açıklaması (Markdown)
- Başlık ve paragraf → Bir sonraki fonksiyonun hangi adımları yaptığına dair özet verir.

## Hücre 10 – Ön işleme bileşenleri ve fonksiyon
- `stop_words = set(stopwords.words('english'))` → Stopword listesini sete çevirerek hızlı üyelik kontrolü sağlar.
- `lemmatizer = WordNetLemmatizer()` → Kelimeleri kök forma döndürmek için nesne oluşturur.
- `stemmer = PorterStemmer()` → Porter stem algoritmasını hazırlayan nesne.
- `punct_pattern = re.compile(r"[^a-zA-Z\s]")` → Harf ve boşluk dışındaki karakterleri hedefleyen regex deseni.
- `def preprocess_text(text: str) -> str:` → Tek bir metni temizleyen fonksiyon tanımı.
- `    # Normalize text through token cleanup, lemmatization, and stemming.` → Yapılan işlemleri hatırlatan yorum satırı.
- `    text = text.lower()` → Tüm karakterleri küçük harfe çevirir (büyük/küçük farkını ortadan kaldırır).
- `    text = punct_pattern.sub(' ', text)` → Regex ile noktalama/özel karakterleri boşlukla değiştirir.
- `    tokens = nltk.word_tokenize(text)` → Metni kelime listesine çevirir (`punkt` paketine ihtiyaç var).
- `    tokens = [token for token in tokens if token.isalpha()]` → Sadece alfabetik kelimeleri tutar, sayıları atar.
- `    tokens = [token for token in tokens if token not in stop_words]` → Stopwordleri çıkararak bilgi taşımayan kelimeleri temizler.
- `    tokens = [lemmatizer.lemmatize(token) for token in tokens]` → Her kelimeyi WordNet lemmatizer ile temel sözlük formuna indirger.
- `    tokens = [stemmer.stem(token) for token in tokens]` → Lemmatize edilmiş kelimeleri Porter stem ile daha da kök haline getirir.
- `    return ' '.join(tokens)` → Temizlenmiş tokenları tekrar stringe dönüştürür (vektörleştirici bu çıktıyı kullanıyor).

## Hücre 11 – Eğitim/test ayırımı
- `X_train_raw, X_test_raw, y_train, y_test = train_test_split(...` → Metinleri ve etiketleri %80 eğitim, %20 test olacak şekilde stratified biçimde böler; rastgelelik için `random_state=42`.
- `print(f'Train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}')` → Bölünmüş setlerin boyutlarını doğrularım.

## Hücre 12 – Ön işleme uygulama süresi
- `%%time` → IPython hücre sihirbazı; hücrenin çalışma süresini raporlar.
- `X_train = [preprocess_text(doc) for doc in X_train_raw]` → Eğitim kümesindeki her belgeyi temizlik fonksiyonundan geçirir.
- `X_test = [preprocess_text(doc) for doc in X_test_raw]` → Test kümesi için aynı işlemleri yapar.

## Hücre 13 – Temizlik öncesi/sonrası karşılaştırma
- `idx = np.random.randint(0, len(X_train))` → Temsili bir örnek seçmek için rastgele indeks üretir.
- `print('Original sample:\n', X_train_raw[idx][:250].strip())` → Temizlik öncesi metnin ilk 250 karakterini basar.
- `print('\nCleaned sample:\n', X_train[idx][:250].strip())` → Aynı metnin temizlenmiş halini gösterir; dönüşümün etkisi görülsün.

## Hücre 14 – Modelleme bölüm başlığı (Markdown)
- Başlık ve paragraf → Farklı vektörleştirici ve sınıflandırıcı kombinasyonlarının deneneceğini açıklar.

## Hücre 15 – Vektörleştirici ve sınıflandırıcı sözlükleri
- `vectorizers = { ... }` → Dört farklı özellik çıkartma yolunu adlandırılmış şekilde saklar.
  - `'Count (1,1)': CountVectorizer(max_features=20000, ngram_range=(1, 1))` → Yalnızca unigram frekanslarını kullanır.
  - `'Count (1,2)': CountVectorizer(max_features=20000, ngram_range=(1, 2))` → Unigram + bigram frekansları.
  - `'TF-IDF (1,1)': TfidfVectorizer(max_features=20000, ngram_range=(1, 1))` → TF-IDF ağırlıklı unigram.
  - `'TF-IDF (1,2)': TfidfVectorizer(max_features=20000, ngram_range=(1, 2))` → TF-IDF ağırlıklı unigram + bigram.
- `classifiers = { ... }` → Üç farklı modelin konfigurasyonunu tutar.
  - `'Multinomial NB': MultinomialNB()` → Varsayılan parametrelerle Naive Bayes.
  - `'Linear SVC': LinearSVC(max_iter=5000)` → Yakınsama garantisi için iterasyon sayısı artırılmış lineer SVM.
  - `'Logistic Regression': LogisticRegression(...)` → `max_iter=5000`, `solver='saga'`, `penalty='l2'`, `multi_class='multinomial'`, `random_state=42` ayarlarıyla çok sınıflı lojistik regresyon; `saga` büyük TF-IDF matrisleriyle uyumlu.

## Hücre 16 – Modelleri eğitip sonuçları toplama
- `results = []` → Her kombinasyon için metrikleri saklayacak liste.
- `best_model = None` → Şu ana kadarki en iyi modeli tutmak için yer tutucu.
- `best_accuracy = 0.0` → En yüksek doğruluk skorunu takip eder.
- `for vec_name, vec in vectorizers.items():` → Her vektörleştirici türü üzerinde döngü başlatır.
- `    fitted_vec = clone(vec)` → Orijinal sözlükteki nesneyi bozmadan yeni bir kopya üretir.
- `    X_train_vec = fitted_vec.fit_transform(X_train)` → Seçilen vektörleştiriciyi eğitip eğitim metinlerini sparse matrise dönüştürür.
- `    X_test_vec = fitted_vec.transform(X_test)` → Aynı vektörleştirici ile test metinlerini dönüştürür (fit yok, sadece transform).
- `    for clf_name, clf in classifiers.items():` → Her sınıflandırıcı için iç döngü.
- `        model = clone(clf)` → Sınıflandırıcının temiz bir kopyasını üretir.
- `        model.fit(X_train_vec, y_train)` → Modeli eğitim verisi üzerinde eğitir.
- `        y_pred = model.predict(X_test_vec)` → Test kümesinin tahminlerini üretir.
- `        accuracy = accuracy_score(y_test, y_pred)` → Doğruluk skorunu hesaplar.
- `        precision, recall, f1, _ = precision_recall_fscore_support(...)` → Sınıf dengesizliklerini dikkate almak için weighted ortalamalı precision/recall/F1 skorlarını hesaplar; `zero_division=0` boş sınıflarda uyarı vermemesi için.
- `        results.append({...})` → Kombinasyonun adı ve metrikleri sözlük halinde kaydedilir.
- `        if accuracy > best_accuracy:` → Mevcut doğruluk daha yüksekse en iyi modeli güncelle.
- `            best_accuracy = accuracy` → En iyi doğruluk güncellemesi.
- `            best_model = {...}` → En iyi modelin adı, eğitilmiş nesneler ve tahminleri saklanır.
- `results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)` → Tüm sonuçlar DataFrame’e dönüştürülür ve doğruluğa göre sıralanır.
- `results_df.reset_index(drop=True, inplace=True)` → Sıralama sonrası index sıfırlanır.
- `results_df` → DataFrame’i notebook çıktısında göstermek için.

## Hücre 17 – Sonuçların görselleştirilmesi
- `results_display = results_df.copy()` → Grafik için ayrı kopya.
- `results_display['Model'] = results_display['Vectorizer'] + ' + ' + results_display['Classifier']` → Kombinasyon adlarını tek sütunda birleştirir.
- `plt.figure(figsize=(10, 6))` → Grafik boyutunu ayarlar.
- `sns.barplot(data=results_display, x='Accuracy', y='Model', palette='viridis')` → Doğruluğu yatay çubuk grafikte gösterir.
- `plt.title('Model accuracy by vectorizer/classifier combination')` → Başlık ekler.
- `plt.xlabel('Accuracy')` ve `plt.ylabel('')` → Eksen etiketleri (y etiketi boş).
- `plt.xlim(0.6, 1.0)` → X eksenini anlamlı aralıkta sınırlar.
- `plt.show()` → Grafiği görüntüler.

## Hücre 18 – En iyi model raporu
- `print(f"Best model: {best_model['vectorizer_name']} + {best_model['classifier_name']}")` → Seçilen vektörleştirici/model çiftini yazar.
- `print(classification_report(y_test, best_model['y_pred'], target_names=target_names))` → Her sınıf için precision/recall/F1 ve destek değerlerini dökümler.

## Hücre 19 – Karışıklık matrisi
- `fig, ax = plt.subplots(figsize=(12, 10))` → Grafik için figür ve eksen oluşturur.
- `ConfusionMatrixDisplay.from_predictions(...)` → Tahminler vs gerçekler için karışıklık matrisi hesaplayıp çizdirir.
- `plt.xticks(rotation=90)` → X ekseni etiketlerini 90 derece döndürüp okunabilir kılar.
- `plt.title('Confusion matrix for best-performing model')` → Başlık ekler.
- `plt.tight_layout()` → Grafiğin üst üste binmesini engeller.
- `plt.show()` → Grafiği ekrana basar.

## Hücre 20 – Yanlış sınıflandırılan örnekleri inceleme
- `misclassified_indices = np.where(best_model['y_pred'] != y_test)[0]` → Tahmin hatası yapılan indeksleri bulur.
- `rng = np.random.default_rng(42)` → NumPy’nin yeni RNG arayüzüyle deterministik rastgelelik sağlar.
- `inspect_count = min(5, len(misclassified_indices))` → En fazla beş örnek göstermek için sayıyı sınırlar.
- `for idx in rng.choice(misclassified_indices, size=inspect_count, replace=False):` → Hatalı örneklerden rastgele seçim yapar.
- `    print('-' * 80)` → Ayraç çizgisi.
- `    print(f"True label: {target_names[y_test[idx]]}")` → Gerçek sınıf adı.
- `    print(f"Predicted label: {target_names[best_model['y_pred'][idx]]}")` → Tahmin edilen sınıf adı.
- `    print('Snippet:', X_test_raw[idx][:400].strip())` → Ham metinden 400 karakterlik özet yazdırır.

## Hücre 21 – Yanlış sınıflandırma yorumları (Markdown)
- Paragraf → Hangi sınıflar arasında karışıklık olduğuna dair kısa değerlendirme.

## Hücre 22 – SVD bölümü (Markdown)
- Başlık ve açıklama → TruncatedSVD’nin neden denendiğini belirtir.

## Hücre 23 – TruncatedSVD pipeline’ı
- `svd_pipeline = Pipeline([...])` → TF-IDF, SVD ve Logistic Regression adımlarını sıraya dizen bir pipeline oluşturur.
  - `('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2)))` → Bigram dahil TF-IDF özellikleri çıkarılır.
  - `('svd', TruncatedSVD(n_components=100, random_state=42))` → 20 000 boyutlu vektörü 100 boyuta indirger.
  - `('clf', LogisticRegression(max_iter=1000, solver='saga', penalty='l2'))` → Düşük boyutlu uzayda lojistik regresyon eğitir.
- `svd_pipeline.fit(X_train, y_train)` → Pipeline’ı eğitim verisi üzerinde fit eder.
- `svd_pred = svd_pipeline.predict(X_test)` → Test verisi için tahmin üretir.
- `svd_accuracy = accuracy_score(y_test, svd_pred)` → Doğruluk hesaplar.
- `svd_precision, svd_recall, svd_f1, _ = precision_recall_fscore_support(...)` → Weighted precision/recall/F1 metriklerini çıkarır.
- `print(f'SVD pipeline accuracy: {svd_accuracy:.3f}')` → Doğruluğu üç ondalıkla yazar.
- `print(f'SVD pipeline weighted F1: {svd_f1:.3f}')` → Weighted F1 değerini üç ondalıkla bildirir.

## Hücre 24 – Grid Search bölümü (Markdown)
- Açıklama → Bir sonraki hücrenin neden hiperparametre araması yaptığına dair kısa bilgi.

## Hücre 25 – GridSearchCV ile lojistik regresyonu ayarlama
- `grid_pipeline = Pipeline([...])` → TF-IDF + Logistic Regression pipeline’ının sade hali.
  - `('tfidf', TfidfVectorizer(max_features=20000))` → TF-IDF çıkarımı.
  - `('clf', LogisticRegression(max_iter=1000, solver='liblinear'))` → Daha küçük veri için hızlı solver.
- `param_grid = {...}` → Denenecek hiperparametre kombinasyonları (n-gram aralığı, max_df ve C).
- `grid_search = GridSearchCV(..., cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)` → 3 katlı çapraz doğrulama ile weighted F1’i optimize eder; `n_jobs=-1` tüm çekirdekleri kullanır, `verbose=1` ilerleme mesajı gösterir.
- `grid_search.fit(X_train, y_train)` → Aramayı başlatır; her kombinasyon için eğitip değerlendirir.
- `print('Best parameters:', grid_search.best_params_)` → En iyi hiperparametre setini bildirir.
- `print(f'Best CV weighted F1: {grid_search.best_score_:.3f}')` → Çapraz doğrulama sonucunu yazar.
- `grid_pred = grid_search.predict(X_test)` → En iyi modelle test verisini tahmin eder.
- `grid_accuracy = accuracy_score(y_test, grid_pred)` → Test doğruluğunu hesaplar.
- `print(f'Test accuracy after tuning: {grid_accuracy:.3f}')` → Tuning sonrası doğruluğu raporlar.

## Hücre 26 – Kısa özet (Markdown)
- Paragraf → En iyi model, ön işlemenin etkisi, SVD ve grid search bulguları hakkında sonuç cümlesi.

---

Bu belge, ödevi savunurken her satırı neden yazdığımı hızlıca hatırlamamı sağlıyor. Notebookta herhangi bir değişiklik yapmadan tüm kodu satır satır açıklamış oldum. Hazırlık sırasında sadece okuma işlemi gerçekleştirdim; defter olduğu gibi kullanılabilir.
