import os, re, json, time, random
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from google_play_scraper import reviews, Sort
from googletrans import Translator

import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers, models, callbacks
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from tabulate import tabulate

nltk.download('punkt')

# ===== KONFIGURASI =====
APP_ID = "com.roblox.client"
TARGET_REVIEWS = 100000
LANG = 'id'
COUNTRY = 'id'
MAX_VOCAB = 25000
MAX_LEN = 100
EPOCHS = 10
BATCH_SIZE = 64
THRESHOLD_AMBIGU = 0.5
DIFF_THRESHOLD = 0.15

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ===== STEMMER =====
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return stemmer.stem(text)

def map_rating(r):
    if r <= 2: return "negatif"
    elif r == 3: return "netral"
    return "positif"


# ==================== HELPER: TABEL PRINT ====================
def print_table(df_or_data, headers=None, title=None):
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    if isinstance(df_or_data, pd.DataFrame):
        print(tabulate(df_or_data, headers='keys', tablefmt='grid',
                       showindex=False, floatfmt=".4f"))
    else:
        print(tabulate(df_or_data, headers=headers or [], tablefmt='grid',
                       floatfmt=".4f"))
    print()


# ================= STEP 1: Pengumpulan Data (Scraping) =================
print(f"[{datetime.now()}] Mulai scraping...")
all_reviews, cursor = [], None

while len(all_reviews) < TARGET_REVIEWS:
    result, cursor = reviews(APP_ID, lang=LANG, country=COUNTRY,
                             sort=Sort.NEWEST, count=200, continuation_token=cursor)
    if not result:
        break
    all_reviews.extend(result)
    print(f"Terambil: {len(all_reviews)}")
    time.sleep(1)

os.makedirs("data", exist_ok=True)
with open("data/reviews_raw.json", "w", encoding="utf-8") as f:
    json.dump(all_reviews, f, ensure_ascii=False, indent=2, default=str)

print(f"✅ Total scraping: {len(all_reviews)}")


# ================= STEP 2: Penerjemahan Data =================
translator = Translator()
rows = []
empty = 0

for r in tqdm(all_reviews):
    text = r.get("content","") or ""
    rating = r.get("score",0)
    if not text.strip():
        empty += 1
        continue
    try:
        translated = translator.translate(text, dest='id').text
    except:
        translated = text

    rows.append({
        "rating": rating,
        "content_raw": text,
        "content_translated": translated,
        "label": map_rating(rating)
    })

df = pd.DataFrame(rows)
df.to_csv("data/reviews_translated.csv", index=False)
print(f"Kosong dibuang: {empty}")
print(f"Data setelah translate: {len(df)}")


# ================= STEP 3: Preprocessing Data =================
df["content_clean"] = df["content_translated"].apply(clean_text)

before = len(df)
df = df.drop_duplicates(subset=["content_clean"])
df = df[df["content_clean"].str.len() > 3]
df.to_csv("data/reviews_clean.csv", index=False)

print(f"Before: {before}, After: {len(df)}")
dist_label = df["label"].value_counts().reset_index()
dist_label.columns = ["label", "jumlah"]
print_table(dist_label, title="Distribusi Label")


# ================= STEP 4: Pelabelan Otomatis =================
label2id = {"negatif":0,"netral":1,"positif":2}
id2label = {v:k for k,v in label2id.items()}

X = df["content_clean"]
y = df["label"].map(label2id)


# ================= STEP 5: Pembagian Data =================
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
)

print(f"Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")

total = len(X_train)+len(X_val)+len(X_test)
split_df = pd.DataFrame([
    ["training", len(X_train), round(len(X_train)/total*100,2)],
    ["validasi", len(X_val),   round(len(X_val)/total*100,2)],
    ["testing",  len(X_test),  round(len(X_test)/total*100,2)],
], columns=["Jenis Data","Jumlah","Persentase (%)"])
split_df.to_csv("data/split_data.csv", index=False)
print_table(split_df, title="Split Data (80/10/10)")


# ================= STEP 6: Pemodelan CNN =================
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

def pad(txt):
    return pad_sequences(tokenizer.texts_to_sequences(txt), maxlen=MAX_LEN)

Xtr, Xva, Xte = pad(X_train), pad(X_val), pad(X_test)

model = models.Sequential([
    layers.Embedding(MAX_VOCAB,128,input_length=MAX_LEN),
    layers.Conv1D(256,3,activation='relu'),
    layers.MaxPooling1D(2),
    layers.Conv1D(128,3,activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(3,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

early = callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(
    Xtr, y_train,
    validation_data=(Xva, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early]
)


# ================= STEP 7: Evaluasi =================
pred = model.predict(Xte)
y_pred = np.argmax(pred,axis=1)

report = classification_report(y_test, y_pred,
                               target_names=list(label2id.keys()),
                               output_dict=True)

# Classification Report dalam tabel
report_rows = []
for kls in list(label2id.keys()) + ["accuracy", "macro avg", "weighted avg"]:
    if kls == "accuracy":
        report_rows.append({
            "Kelas":     "accuracy",
            "Precision": "",
            "Recall":    "",
            "F1-Score":  round(report["accuracy"], 4),
            "Support":   "",
        })
    else:
        r = report[kls]
        report_rows.append({
            "Kelas":     kls,
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"], 4),
            "F1-Score":  round(r["f1-score"], 4),
            "Support":   int(r["support"]) if "support" in r else "",
        })

report_df = pd.DataFrame(report_rows)
report_df.to_csv("data/classification_report.csv", index=False)
print_table(report_df, title="Classification Report")

# Summary Metrics dalam tabel
summary_df = pd.DataFrame([
    ["Accuracy",      round(report["accuracy"], 4)],
    ["Precision Avg", round(report["weighted avg"]["precision"], 4)],
    ["Recall Avg",    round(report["weighted avg"]["recall"], 4)],
    ["F1-Score Avg",  round(report["weighted avg"]["f1-score"], 4)],
    ["F1 Negatif",    round(report["negatif"]["f1-score"], 4)],
    ["F1 Netral",     round(report["netral"]["f1-score"], 4)],
    ["F1 Positif",    round(report["positif"]["f1-score"], 4)],
], columns=["Metrik", "Nilai"])
summary_df.to_csv("data/summary_metrics.csv", index=False)
print_table(summary_df, title="Ringkasan Metrik Evaluasi")

# Confusion Matrix dalam tabel
cm = confusion_matrix(y_test, y_pred)
labels = list(label2id.keys())
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
cm_df.index.name = "Aktual \\ Prediksi"
cm_df.to_csv("data/confusion_matrix.csv")
cm_display = cm_df.reset_index()
print_table(cm_display, title="Confusion Matrix")

# Simpan juga file evaluasi lengkap
report_full = pd.DataFrame(report).transpose().reset_index()
report_full.columns = ["kelas","precision","recall","f1_score","support"]
report_full.to_csv("data/evaluasi.csv", index=False)

# Plot learning curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='train acc', marker='o')
plt.plot(history.history['val_accuracy'], label='val acc',   marker='s')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='train loss', marker='o')
plt.plot(history.history['val_loss'], label='val loss',   marker='s')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("data/learning_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("📊 Learning curve disimpan di data/learning_curve.png")

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=labels, yticklabels=labels,
    linewidths=0.5, linecolor='gray',
    cbar_kws={'label': 'Jumlah'}, ax=ax
)
ax.set_xlabel('Prediksi', fontsize=12, labelpad=10)
ax.set_ylabel('Aktual', fontsize=12, labelpad=10)
ax.set_title('Confusion Matrix — CNN Analisis Sentimen', fontsize=13, pad=14)
plt.tight_layout()
plt.savefig("data/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("📊 Confusion matrix disimpan di data/confusion_matrix.png")

# Plot Distribusi Label
dist_final = df["label"].value_counts().reindex(["positif","netral","negatif"]).reset_index()
dist_final.columns = ["label","jumlah"]
colors = {"positif":"#4CAF50","netral":"#2196F3","negatif":"#F44336"}
bar_colors = [colors[l] for l in dist_final["label"]]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(dist_final["label"], dist_final["jumlah"],
              color=bar_colors, edgecolor='white', linewidth=0.8)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h+50,
            f'{int(h):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Distribusi Data — Positif / Netral / Negatif', fontsize=13, pad=12)
ax.set_xlabel('Kelas Sentimen', fontsize=11)
ax.set_ylabel('Jumlah Data', fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("data/distribusi_label.png", dpi=150, bbox_inches='tight')
plt.close()
print("📊 Distribusi label disimpan di data/distribusi_label.png")


# ================= STEP 8: Deteksi Ambiguitas =================
all_probs = model.predict(pad(df["content_clean"]))

rows_pred = []
for i, p in enumerate(all_probs):
    top = np.argmax(p)
    conf = p[top]
    diff = np.sort(p)[-1] - np.sort(p)[-2]

    label = id2label[top]
    status = "tidak ambigu"

    if conf < THRESHOLD_AMBIGU or diff < DIFF_THRESHOLD:
        label = "Ambigu"
        status = "ambigu"

    rows_pred.append({
        "text":       df.iloc[i]["content_clean"],
        "true":       df.iloc[i]["label"],
        "pred":       label,
        "confidence": round(float(conf), 4),
        "status":     status
    })

df_pred = pd.DataFrame(rows_pred)
df_pred.to_csv("data/prediksi_all.csv", index=False)

ambigu_summary = df_pred["status"].value_counts().reset_index()
ambigu_summary.columns = ["Kategori","Jumlah"]
ambigu_summary.to_csv("data/ambigu_summary.csv", index=False)
print_table(ambigu_summary, title="Ringkasan Status Ambiguitas")

# Plot Distribusi Ambigu
colors_ambigu = {"tidak ambigu":"#4CAF50","ambigu":"#FF9800"}
ambigu_colors = [colors_ambigu.get(k,"#999999") for k in ambigu_summary["Kategori"]]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(ambigu_summary["Kategori"], ambigu_summary["Jumlah"],
              color=ambigu_colors, edgecolor='white', linewidth=0.8)
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h+50,
            f'{int(h):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_title('Distribusi Status Ambiguitas', fontsize=13, pad=12)
ax.set_xlabel('Status', fontsize=11)
ax.set_ylabel('Jumlah Data', fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig("data/distribusi_ambigu.png", dpi=150, bbox_inches='tight')
plt.close()
print("📊 Distribusi ambigu disimpan di data/distribusi_ambigu.png")

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
print("\n✅ SELESAI")