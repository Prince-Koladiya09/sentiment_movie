import os, json, time, pickle, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path

# TensorFlow / Keras Fix for 2.16+ compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Multiprocess Fix for Python 3.12 (handles shutdown lock issue)
try:
    import multiprocess.resource_tracker
    orig_stop_locked = multiprocess.resource_tracker.ResourceTracker._stop_locked
    
    def patched_stop_locked(self, *args, **kwargs):
        if hasattr(self._lock, "_recursion_count"):
            return orig_stop_locked(self, *args, **kwargs)
        if self._fd is None or self._pid is None: return
        try:
            import os
            os.close(self._fd)
            self._fd = None
            os.waitpid(self._pid, 0)
            self._pid = None
        except Exception:
            pass

    multiprocess.resource_tracker.ResourceTracker._stop_locked = patched_stop_locked
except Exception:
    pass

warnings.filterwarnings("ignore")

EXPORT_DIR = Path("backend/data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
Path("backend/models/saved").mkdir(parents=True, exist_ok=True)

import nltk
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text   = re.sub(r"<.*?>", "", text)
    text   = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    tokens = [lemmatizer.lemmatize(w) for w in text.split()
              if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)

print("Loading IMDB dataset...")
try:
    df = pd.read_csv("IMDB Dataset.csv")
    df.columns = ["review", "sentiment"]
    df["label"] = (df["sentiment"] == "positive").astype(int)
    print(f"  Loaded from CSV: {len(df)} reviews")
except FileNotFoundError:
    try:
        from datasets import load_dataset
        raw      = load_dataset("imdb")
        train_df = pd.DataFrame(raw["train"])
        test_df  = pd.DataFrame(raw["test"])
        df       = pd.concat([train_df, test_df], ignore_index=True)
        df.rename(columns={"text": "review"}, inplace=True)
        df["sentiment"] = df["label"].map({1: "positive", 0: "negative"})
        print(f"  Downloaded via HuggingFace datasets: {len(df)} reviews")
    except Exception as e:
        print(f"\nERROR: Could not load IMDB dataset: {e}")
        raise SystemExit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                               f1_score, roc_auc_score, confusion_matrix,
                               roc_curve, precision_recall_curve)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

print("Preprocessing text...")
train_df = train_df.copy()
test_df  = test_df.copy()
train_df["clean"] = train_df["review"].apply(preprocess)
test_df["clean"]  = test_df["review"].apply(preprocess)

X_train_raw = train_df["review"].tolist()
X_test_raw  = test_df["review"].tolist()
X_train     = train_df["clean"].tolist()
X_test      = test_df["clean"].tolist()
y_train     = train_df["label"].tolist()
y_test      = test_df["label"].tolist()

all_metrics, all_cm, all_roc, all_pr_curves = {}, {}, {}, {}
all_errors, all_conf_dist, all_feature_imp, all_model_preds = {}, {}, {}, {}

def compute_metrics(name, y_true, y_pred, y_proba, ms):
    y_true = list(y_true); y_pred = list(y_pred); y_proba = list(y_proba)
    acc    = float(accuracy_score(y_true, y_pred))
    prec   = float(precision_score(y_true, y_pred, zero_division=0))
    rec    = float(recall_score(y_true, y_pred, zero_division=0))
    f1     = float(f1_score(y_true, y_pred, zero_division=0))
    auc    = float(roc_auc_score(y_true, y_proba))
    fpr, tpr, _      = roc_curve(y_true, y_proba)
    prec_c, rec_c, _ = precision_recall_curve(y_true, y_proba)
    step  = max(1, len(fpr) // 200)
    step2 = max(1, len(prec_c) // 200)
    all_metrics[name]   = {"Accuracy": round(acc,4), "Precision": round(prec,4),
                            "Recall": round(rec,4), "F1-Score": round(f1,4),
                            "AUC-ROC": round(auc,4), "Inference_ms": round(ms,4)}
    all_cm[name]        = confusion_matrix(y_true, y_pred).tolist()
    all_roc[name]       = [{"fpr": round(float(f),4), "tpr": round(float(t),4)}
                            for f,t in zip(fpr[::step], tpr[::step])]
    all_pr_curves[name] = [{"precision": round(float(p),4), "recall": round(float(r),4)}
                            for p,r in zip(prec_c[::step2], rec_c[::step2])]
    print(f"  {name}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}  ({ms:.3f}ms/sample)")

def collect_errors(name, y_true, y_pred, y_proba, texts, max_errors=60):
    errors = []
    for i, (yt, yp, prob) in enumerate(zip(y_true, y_pred, y_proba)):
        if int(yt) != int(yp):
            conf = float(max(prob, 1.0 - prob))
            errors.append({"id": i, "model": name, "review": texts[i][:400],
                           "true_label": "positive" if int(yt)==1 else "negative",
                           "pred_label": "positive" if int(yp)==1 else "negative",
                           "confidence": round(conf,4), "length": len(texts[i]),
                           "error_type": "False Positive" if int(yp)==1 else "False Negative"})
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    all_errors[name] = errors[:max_errors]

def collect_confidence_dist(name, y_true, y_pred, y_proba, bins=20):
    edges   = np.linspace(0.5, 1.0, bins+1)
    correct = [float(max(p,1-p)) for yt,yp,p in zip(y_true,y_pred,y_proba) if int(yt)==int(yp)]
    wrong   = [float(max(p,1-p)) for yt,yp,p in zip(y_true,y_pred,y_proba) if int(yt)!=int(yp)]
    cc, _   = np.histogram(correct, bins=edges)
    wc, _   = np.histogram(wrong,   bins=edges)
    all_conf_dist[name] = [{"bucket": f"{edges[i]:.2f}-{edges[i+1]:.2f}",
                             "correct": int(cc[i]), "wrong": int(wc[i])} for i in range(bins)]

print("\nTraining Naive Bayes...")
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nb_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=3)),
    ("clf",   MultinomialNB(alpha=0.1)),
])
nb_pipe.fit(X_train, y_train)
t0       = time.time()
nb_pred  = nb_pipe.predict(X_test)
nb_proba = nb_pipe.predict_proba(X_test)[:, 1]
ms       = (time.time()-t0)*1000/len(X_test)
compute_metrics("Naive Bayes", y_test, nb_pred, nb_proba, ms)
collect_errors("Naive Bayes", y_test, nb_pred, nb_proba, X_test_raw)
collect_confidence_dist("Naive Bayes", y_test, nb_pred, nb_proba)
all_model_preds["Naive Bayes"] = nb_pred.tolist()

nb_feats   = nb_pipe.named_steps["tfidf"].get_feature_names_out()
nb_lp      = nb_pipe.named_steps["clf"].feature_log_prob_
nb_pos_idx = np.argsort(nb_lp[1])[-30:][::-1]
nb_neg_idx = np.argsort(nb_lp[0])[-30:][::-1]
all_feature_imp["Naive Bayes"] = {
    "positive": [{"word": nb_feats[i], "weight": round(float(nb_lp[1][i]),4)} for i in nb_pos_idx],
    "negative": [{"word": nb_feats[i], "weight": round(float(nb_lp[0][i]),4)} for i in nb_neg_idx],
}
with open("backend/models/saved/naive_bayes_pipeline.pkl","wb") as f:
    pickle.dump(nb_pipe, f)

print("\nTraining Logistic Regression...")
from sklearn.linear_model import LogisticRegression
lr_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=2, sublinear_tf=True)),
    ("clf",   LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs", n_jobs=-1)),
])
lr_pipe.fit(X_train, y_train)
t0       = time.time()
lr_pred  = lr_pipe.predict(X_test)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
ms       = (time.time()-t0)*1000/len(X_test)
compute_metrics("Logistic Regression", y_test, lr_pred, lr_proba, ms)
collect_errors("Logistic Regression", y_test, lr_pred, lr_proba, X_test_raw)
collect_confidence_dist("Logistic Regression", y_test, lr_pred, lr_proba)
all_model_preds["Logistic Regression"] = lr_pred.tolist()

lr_feats   = lr_pipe.named_steps["tfidf"].get_feature_names_out()
lr_coefs   = lr_pipe.named_steps["clf"].coef_[0]
lr_pos_idx = np.argsort(lr_coefs)[-30:][::-1]
lr_neg_idx = np.argsort(lr_coefs)[:30]
all_feature_imp["Logistic Regression"] = {
    "positive": [{"word": lr_feats[i], "weight": round(float(lr_coefs[i]),4)} for i in lr_pos_idx],
    "negative": [{"word": lr_feats[i], "weight": round(float(lr_coefs[i]),4)} for i in lr_neg_idx],
}
with open("backend/models/saved/logistic_regression_pipeline.pkl","wb") as f:
    pickle.dump(lr_pipe, f)

print("\nTraining Bidirectional LSTM...")
lstm_history = None
try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras import layers, Model

    VOCAB_SIZE = 20000; MAX_LEN = 200; EMBED_DIM = 64; EPOCHS = 8
    kt = KerasTokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    kt.fit_on_texts(X_train)
    X_tr_seq = pad_sequences(kt.texts_to_sequences(X_train), maxlen=MAX_LEN, truncating="post", padding="post")
    X_te_seq = pad_sequences(kt.texts_to_sequences(X_test),  maxlen=MAX_LEN, truncating="post", padding="post")
    y_tr_arr = np.array(y_train); y_te_arr = np.array(y_test)

    inp  = layers.Input(shape=(MAX_LEN,))
    x    = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(inp)
    x    = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x    = layers.Bidirectional(layers.LSTM(32))(x)
    x    = layers.Dense(32, activation="relu")(x)
    x    = layers.Dropout(0.4)(x)
    out  = layers.Dense(1, activation="sigmoid")(x)
    lstm_model = Model(inp, out)
    lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    cb   = tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
    hist = lstm_model.fit(X_tr_seq, y_tr_arr, epochs=EPOCHS, batch_size=256,
                          validation_split=0.1, callbacks=[cb], verbose=1)

    lstm_history = {"rnn_lstm": [
        {"epoch": i+1,
         "train_acc":  round(float(hist.history["accuracy"][i]),4),
         "val_acc":    round(float(hist.history["val_accuracy"][i]),4),
         "train_loss": round(float(hist.history["loss"][i]),4),
         "val_loss":   round(float(hist.history["val_loss"][i]),4)}
        for i in range(len(hist.history["accuracy"]))
    ]}

    t0         = time.time()
    lstm_proba = lstm_model.predict(X_te_seq, verbose=0).flatten()
    ms         = (time.time()-t0)*1000/len(X_test)
    lstm_pred  = (lstm_proba > 0.5).astype(int)

    compute_metrics("RNN (LSTM)", y_test, lstm_pred, lstm_proba, ms)
    collect_errors("RNN (LSTM)", y_test, lstm_pred, lstm_proba, X_test_raw)
    collect_confidence_dist("RNN (LSTM)", y_test, lstm_pred, lstm_proba)
    all_model_preds["RNN (LSTM)"] = lstm_pred.tolist()

    lstm_model.save("backend/models/saved/rnn_lstm.keras")
    with open("backend/models/saved/lstm_tokenizer.pkl","wb") as f:
        pickle.dump(kt, f)

except ImportError:
    print("  TensorFlow not installed — skipping LSTM.")

print("\nTraining DistilBERT...")
bert_history = None
try:
    import tensorflow as tf
    from transformers import (DistilBertTokenizerFast,
                               TFDistilBertForSequenceClassification,
                               create_optimizer)

    BERT_MAX_LEN = 128; BERT_BATCH = 32; BERT_EPOCHS = 3
    bert_tok = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def encode(texts, max_len):
        enc = bert_tok(list(texts), truncation=True, padding="max_length",
                       max_length=max_len, return_tensors="tf")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    n_train = min(len(X_train), 20000); n_test = min(len(X_test), 5000)
    tr_enc  = encode(X_train_raw[:n_train], BERT_MAX_LEN)
    te_enc  = encode(X_test_raw[:n_test],   BERT_MAX_LEN)
    y_tr_b  = np.array(y_train[:n_train]); y_te_b = np.array(y_test[:n_test])

    bert_model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2)
    steps  = (n_train // BERT_BATCH) * BERT_EPOCHS
    opt, _ = create_optimizer(init_lr=2e-5, num_warmup_steps=steps//10, num_train_steps=steps)
    bert_model.compile(optimizer=opt,
                        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics=["accuracy"])

    train_ds = (tf.data.Dataset.from_tensor_slices(
        ({"input_ids": tr_enc["input_ids"], "attention_mask": tr_enc["attention_mask"]}, y_tr_b))
        .shuffle(1000).batch(BERT_BATCH).prefetch(tf.data.AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices(
        ({"input_ids": te_enc["input_ids"], "attention_mask": te_enc["attention_mask"]}, y_te_b))
        .batch(BERT_BATCH).prefetch(tf.data.AUTOTUNE))

    bert_hist_obj = bert_model.fit(train_ds, validation_data=val_ds, epochs=BERT_EPOCHS, verbose=1)

    bert_history = {"distilbert": [
        {"epoch": i+1,
            "train_acc":  round(float(bert_hist_obj.history["accuracy"][i]),4),
            "val_acc":    round(float(bert_hist_obj.history["val_accuracy"][i]),4),
            "train_loss": round(float(bert_hist_obj.history["loss"][i]),4),
            "val_loss":   round(float(bert_hist_obj.history["val_loss"][i]),4)}
        for i in range(BERT_EPOCHS)
    ]}

    t0         = time.time()
    logits     = bert_model.predict(val_ds, verbose=0).logits
    ms         = (time.time()-t0)*1000/n_test
    bert_proba = tf.nn.softmax(logits, axis=-1).numpy()[:, 1]
    bert_pred  = (bert_proba > 0.5).astype(int)

    compute_metrics("DistilBERT", y_te_b.tolist(), bert_pred.tolist(), bert_proba.tolist(), ms)
    collect_errors("DistilBERT", y_te_b.tolist(), bert_pred.tolist(), bert_proba.tolist(), X_test_raw[:n_test])
    collect_confidence_dist("DistilBERT", y_te_b.tolist(), bert_pred.tolist(), bert_proba.tolist())
    all_model_preds["DistilBERT"] = bert_pred.tolist()

    bert_model.save_pretrained("backend/models/saved/distilbert")
    bert_tok.save_pretrained("backend/models/saved/distilbert")

except ImportError:
    print("  transformers / tensorflow not installed — skipping DistilBERT.")

print("\nGenerating LIME explanations...")
lime_results = []
try:
    import lime.lime_text
    easy_pos = [i for i,(yt,yp,ypr) in enumerate(zip(y_test,nb_pred,nb_proba)) if yt==1 and yp==1 and ypr>0.92][:4]
    easy_neg = [i for i,(yt,yp,ypr) in enumerate(zip(y_test,nb_pred,nb_proba)) if yt==0 and yp==0 and ypr<0.08][:4]
    hard     = [i for i,(yt,yp) in enumerate(zip(y_test,nb_pred)) if yt!=yp][:4]
    sample_indices = easy_pos + easy_neg + hard

    for model_name, predict_fn in [
        ("Naive Bayes",         lambda texts: nb_pipe.predict_proba([preprocess(t) for t in texts])),
        ("Logistic Regression", lambda texts: lr_pipe.predict_proba([preprocess(t) for t in texts])),
    ]:
        explainer = lime.lime_text.LimeTextExplainer(class_names=["negative","positive"])
        for idx in sample_indices:
            try:
                exp        = explainer.explain_instance(X_test_raw[idx], predict_fn, num_features=10, num_samples=500)
                true_label = "positive" if y_test[idx]==1 else "negative"
                pred_label = "positive" if nb_pred[idx]==1 else "negative"
                conf       = float(nb_proba[idx]) if pred_label=="positive" else float(1-nb_proba[idx])
                lime_results.append({
                    "id": len(lime_results), "model": model_name, "text": X_test_raw[idx][:500],
                    "true_label": true_label, "pred_label": pred_label, "correct": true_label==pred_label,
                    "confidence": round(conf,4), "words": [{"word": w, "weight": round(float(wt),4)} for w,wt in exp.as_list()],
                })
            except Exception as ex:
                print(f"    LIME failed for sample {idx}: {ex}")
except ImportError:
    print("  lime not installed — skipping.")

print("\nComputing dataset statistics...")
df["length"] = df["review"].str.len()
pos_df = df[df["label"]==1]; neg_df = df[df["label"]==0]
bins = list(range(0, 2400, 200))
pos_hist, edges = np.histogram(pos_df["length"].clip(upper=2200), bins=bins)
neg_hist, _     = np.histogram(neg_df["length"].clip(upper=2200), bins=bins)
length_dist = [{"bucket": f"{int(edges[i])}-{int(edges[i+1])}", "positive": int(pos_hist[i]), "negative": int(neg_hist[i])} for i in range(len(pos_hist))]

len_acc_data = []
for lo, hi in [(0,300),(300,600),(600,1000),(1000,1500),(1500,99999)]:
    mask = [lo <= len(t) < hi for t in X_test_raw]
    yt_b = [y for y,m in zip(y_test,mask) if m]
    yp_b = [y for y,m in zip(lr_pred,mask) if m]
    if yt_b:
        len_acc_data.append({"bucket": f"{lo}-{hi}" if hi<99999 else f"{lo}+", "count": len(yt_b), "accuracy": round(float(accuracy_score(yt_b,yp_b)),4)})

lr_feats_out = lr_pipe.named_steps["tfidf"].get_feature_names_out()
coefs        = lr_pipe.named_steps["clf"].coef_[0]
top_pos_idx  = np.argsort(coefs)[-20:][::-1]; top_neg_idx  = np.argsort(coefs)[:20]
top_pos_words = [{"word": lr_feats_out[i], "count": round(abs(float(coefs[i]))*1000,1)} for i in top_pos_idx]
top_neg_words = [{"word": lr_feats_out[i], "count": round(abs(float(coefs[i]))*1000,1)} for i in top_neg_idx]

sample_reviews = []
for i in range(min(100, len(test_df))):
    r = test_df.iloc[i]
    sample_reviews.append({"id": i+1, "text": str(r["review"])[:300], "sentiment": int(r["label"]), "length": len(str(r["review"]))})

dataset_stats = {
    "total_reviews": len(df), "train_size": len(train_df), "test_size": len(test_df),
    "length_distribution": length_dist,
    "length_stats": {
        "overall":  {"mean": round(float(df["length"].mean()),1), "median": round(float(df["length"].median()),1), "std": round(float(df["length"].std()),1)},
        "positive": {"mean": round(float(pos_df["length"].mean()),1), "median": round(float(pos_df["length"].median()),1)},
        "negative": {"mean": round(float(neg_df["length"].mean()),1), "median": round(float(neg_df["length"].median()),1)},
    },
    "sentiment_distribution": [
        {"name": "Positive", "value": int((df["label"]==1).sum()), "color": "#81b29a"},
        {"name": "Negative", "value": int((df["label"]==0).sum()), "color": "#e05c8c"},
    ],
    "top_positive_words":  top_pos_words, "top_negative_words":  top_neg_words,
    "length_vs_accuracy":  len_acc_data, "sample_reviews":      sample_reviews,
}

print("Computing model agreement matrix...")
model_names = list(all_model_preds.keys())
agreement   = {}
for m1 in model_names:
    agreement[m1] = {}
    for m2 in model_names:
        p1 = np.array(all_model_preds[m1]); p2 = np.array(all_model_preds[m2])
        n  = min(len(p1), len(p2))
        agreement[m1][m2] = round(float(np.mean(p1[:n]==p2[:n])),4)

min_len   = min(len(v) for v in all_model_preds.values())
all_wrong = []
for i in range(min_len):
    preds = [all_model_preds[m][i] for m in model_names]
    if all(int(p) != int(y_test[i]) for p in preds):
        all_wrong.append({"idx": i, "review": X_test_raw[i][:300], "true_label": "positive" if y_test[i]==1 else "negative"})
    if len(all_wrong) >= 20: break

training_history = {}
try:
    if lstm_history: training_history.update(lstm_history)
    if bert_history: training_history.update(bert_history)
except NameError: pass

print("\nSaving exports...")
exports = {
    "metrics.json": all_metrics, "confusion_matrices.json": all_cm, "roc_curves.json": all_roc,
    "pr_curves.json": all_pr_curves, "training_history.json": training_history,
    "error_samples.json": all_errors, "lime_examples.json": lime_results,
    "confidence_dist.json": all_conf_dist, "feature_importance.json": all_feature_imp,
    "dataset_stats.json": dataset_stats, "model_agreement.json": {"matrix": agreement, "all_wrong": all_wrong},
}
for fname, data in exports.items():
    path = EXPORT_DIR / fname
    path.write_text(json.dumps(data, indent=2))

print("\nTraining and export complete!")
print(f"Files saved to: {EXPORT_DIR.resolve()}")
for name, m in all_metrics.items():
    print(f"  {name:25s}  Acc={m['Accuracy']:.4f}  F1={m['F1-Score']:.4f}")
