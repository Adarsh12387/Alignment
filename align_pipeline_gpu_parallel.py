# align_pipeline_gpu_parallel.py
# Parallelized pipeline (Option A) with optional multi-GPU servers.
# Preserves original functionality. Use --run_mode parallel to enable.

import os
import re
import unicodedata
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from langdetect import detect_langs
from scipy.spatial.distance import cosine
from simalign import SentenceAligner
from collections import defaultdict
import stanza
import multiprocessing as mp
import time
import uuid

# ------------------------------- User-tunable defaults -------------------------------
DEFAULT_BATCH_SIZE = 64  # embedding batch size used by GPU server
EMBED_CACHE = {}  # per-process small cache for sequential mode (kept)
# ------------------------------------------------------------------------------------

# 🔐 Global semaphore to limit concurrent GPU embedding jobs
# This is the key protection against GPU OOM when multiple workers are active.
GPU_SEMAPHORE = mp.Semaphore(1)  # allow only 1 concurrent embedding job per GPU

# Device info
MAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Main device (launcher): {MAIN_DEVICE}")

# Keep your existing flags
USE_SONAR = True
USE_SIMALIGN = True

# ---------------------- language maps (unchanged) ----------------------
language_codes = {
    "English": "eng", "Hindi": "hin", "Urdu": "urd-script_arabic",
    "Assamese": "asm", "Bengali": "ben", "Marathi": "mar", "Tamil": "tam",
    "Telugu": "tel", "Kannada": "kan", "Malayalam": "mal", "Punjabi": "pan",
    "Manipuri": "ben","Gujarathi":"guj","Odia": "ory",
    "Bodo": "asm", "Chattisgarhi": "hin", "Dogri": "hin",
    "Garo": "bod", "Galo": "bod", "Jaintia": "bod", "Kashmiri": "urd-script_arabic",
    "Khasi": "bod", "Kokborok": "ben", "Konkani": "hin", "Ladakhi": "bod",
    "Lepcha": "bod", "Maithili": "hin", "Mizo": "bod", "Nepali": "hin",
    "Purgi": "urd-script_arabic", "Sanskrit": "hin", "Santhali": "ben",
    "Sargujia": "hin", "Sikkimese": "bod", "Sindhi": "hin",
}

sonar_language_codes = {
    "English": "eng_Latn", "Hindi": "hin_Deva", "Urdu": "urd_Arab",
    "Assamese": "asm_Beng", "Bengali": "ben_Beng", "Marathi": "mar_Deva",
    "Tamil": "tam_Taml", "Telugu": "tel_Telu", "Kannada": "kan_Knda",
    "Malayalam": "mal_Mlym", "Punjabi": "pan_Guru","Manipuri":"ben_Beng",
    "Gujarathi": "guj_Gujr","Odia": "ory_Orya",
    "Bodo": "asm_Beng", "Chattisgarhi": "hin_Deva", "Dogri": "hin_Deva",
    "Garo": "bod_Tibt", "Galo": "bod_Tibt", "Jaintia": "bod_Tibt", "Kashmiri": "urd_Arab",
    "Khasi": "bod_Tibt", "Kokborok": "ben_Beng", "Konkani": "hin_Deva", "Ladakhi": "bod_Tibt",
    "Lepcha": "bod_Tibt", "Maithili": "hin_Deva", "Mizo": "bod_Tibt", "Nepali": "hin_Deva",
    "Purgi": "urd_Arab", "Sanskrit": "hin_Deva", "Santhali": "ben_Beng",
    "Sargujia": "hin_Deva", "Sikkimese": "bod_Tibt", "Sindhi": "hin_Deva",
}

detect_lang = {
    "eng": "en", "hin": "hi", "urd-script_arabic": "ur",
    "asm": "bn", "ben": "bn", "mar": "mr", "tam": "ta",
    "tel": "te", "kan": "multilingual", "mal": "ml", "pan": "multilingual",
    "mani":"bn","guj":"multilingual","ory":"or","bod":"multilingual"
}

# ---------------------- Utilities and normalization (unchanged) ----------------------
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def normalize_unicode(text):
    return unicodedata.normalize("NFC", text)

def strip_control_chars(text):
    return re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

def normalize_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_punctuation(text):
    replacements = {
        "“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-",
        "…": "...", "ـ": "", "•": ".", "·": "."
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def normalize_digits(text):
    return "".join(str(unicodedata.digit(ch)) if ch.isdigit() else ch for ch in text)

def normalize_arabic_script(text):
    replacements = {
        "ى": "ي", "يٰ": "ي", "ة": "ه", "ۀ": "ه",
        "أ": "ا", "إ": "ا", "آ": "ا",
        "ً": "", "ٌ": "", "ٍ": "", "َ": "", "ُ": "", "ِ": "", "ّ": "", "ْ": ""
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = text.replace("ـ", "")
    return text

def normalize_indic_script(text):
    nukta_map = {
        "ड़": "ड", "ढ़": "ढ",
        "क़": "क", "ख़": "ख", "ग़": "ग",
        "ज़": "ज", "फ़": "फ", "य़": "य"
    }
    for k, v in nukta_map.items():
        text = text.replace(k, v)
    text = text.replace("ँ", "ं")
    return text

def normalize_assamese(text):
    text = re.sub(r"(?<![অ-হয়ৰৱ])([ািীুূৃেৈোৌ])", r"", text)
    text = re.sub(r"([োো])\s+([োো])", r"\1", text)
    text = re.sub(r"([অ-হয়ৰৱ])\s+([া-ৌেো])", r"\1\2", text)
    text = re.sub(r"^[া-ৌেো]+" , "", text)
    text = re.sub(r"([়])\1+", r"\1", text)
    text = text.replace("্ী", "ী")
    text = text.replace("  ", " ")
    return text

def normalize_text_new(text, lang_code):
    text = normalize_unicode(text)
    text = strip_control_chars(text)
    text = normalize_punctuation(text)
    text = normalize_digits(text)
    text = normalize_whitespace(text)

    if "urd" in lang_code or "arabic" in lang_code:
        text = normalize_arabic_script(text)
    elif lang_code == "asm" or lang_code == "ben":
        text = normalize_assamese(text)
    elif lang_code in {"hin","mar","tam","tel","kan","mal","pan","guj","ory"}:
        text = normalize_indic_script(text)

    text = normalize_whitespace(text)
    return text

# Splitting (use stanza as before)
nlp_pipelines = {}
def get_nlp_pipeline(lang_code):
    if lang_code not in nlp_pipelines:
        nlp_pipelines[lang_code] = stanza.Pipeline(
            lang=detect_lang[lang_code],
            processors='tokenize',
            tokenize_no_ssplit=False,
            use_gpu=torch.cuda.is_available()
        )
    return nlp_pipelines[lang_code]

def split_sentences(text, lang_code=None, min_len=0, use_buffer=False, normalize=False):
    sentences = []
    if lang_code:
        try:
            nlp = get_nlp_pipeline(lang_code)
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sentences if sent.text.strip()]
        except Exception as e:
            tqdm.write(f"⚠️ Stanza splitting failed: {e}")
            sentences = [text.strip()]
    else:
        sentences = [text.strip()]

    final_sentences = []
    sentence_enders = r'[-।?؟॥።।܀۔⁇⁈⁉‼⸮⸼…—\~\n\r\t\.|]'
    pattern = f'({sentence_enders}+)\\s*'

    for sent in sentences:
        parts = re.split(pattern, sent)
        combined = [(parts[i] + parts[i + 1]).strip() for i in range(0, len(parts) - 1, 2)]
        if len(parts) % 2 == 1:
            combined.append(parts[-1].strip())

        buffer = ""
        for s in combined:
            candidate = s.strip()
            if normalize:
                candidate = normalize_text_new(candidate, lang_code)

            if use_buffer:
                if len(candidate.split()) <= min_len or re.fullmatch(r'[\W_]+', candidate):
                    buffer += " " + candidate
                else:
                    if buffer:
                        if final_sentences:
                            final_sentences[-1] += buffer
                        else:
                            final_sentences.append(buffer.strip())
                        buffer = ""
                    final_sentences.append(candidate)
            else:
                if len(candidate.split()) > min_len and not re.fullmatch(r'[\W_]+', candidate):
                    final_sentences.append(candidate)

        if use_buffer and buffer:
            if final_sentences:
                final_sentences[-1] += buffer
            else:
                final_sentences.append(buffer.strip())

    return [s for s in final_sentences if s.strip()]

def filter_language_mismatch(sentences, lang):
    filtered = []
    for s in sentences:
        try:
            detected = detect_langs(s)[0].lang
            if lang == "English" and detected != "en":
                continue
            if lang != "English" and detected == "en":
                continue
        except:
            pass
        filtered.append(s)
    return filtered

# mean_pooling (unchanged)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

# --------------------------------------------------------------------------------
# Sequential (single-process) model loading for local runs (same as original)
# --------------------------------------------------------------------------------
if USE_SONAR:
    print("🔹 Loading Sonar (sequential) ...")
    text_embedder_seq = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=MAIN_DEVICE
    )

labse_model_seq = SentenceTransformer("sentence-transformers/LaBSE", device=str(MAIN_DEVICE))

from transformers import AutoTokenizer, AutoModel
indic_tokenizer_seq = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
indic_model_seq = AutoModel.from_pretrained("ai4bharat/indic-bert").to(MAIN_DEVICE)

gte_tokenizer_seq = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
gte_model_seq = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(MAIN_DEVICE)

bge_tokenizer_seq = AutoTokenizer.from_pretrained("BAAI/bge-m3")
bge_model_seq = AutoModel.from_pretrained("BAAI/bge-m3").to(MAIN_DEVICE)

if USE_SIMALIGN:
    print("🔹 Loading SimAlign (sequential) ...")
    aligner_seq = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

print("[INFO] Sequential models loaded.")

# -------------------------------------------------------------------------
# GPU server code: runs in its own process and loads models on assigned GPU
# -------------------------------------------------------------------------
def load_models_on_device(device_id):
    """Load the same set of models on the specified CUDA device and return as dict."""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    models = {}
    if USE_SONAR:
        models["text_embedder"] = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device
        )
    models["labse"] = SentenceTransformer("sentence-transformers/LaBSE", device=str(device))
    from transformers import AutoTokenizer, AutoModel
    models["indic_tokenizer"] = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
    models["indic_model"] = AutoModel.from_pretrained("ai4bharat/indic-bert").to(device)
    models["gte_tokenizer"] = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    models["gte_model"] = AutoModel.from_pretrained("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True).to(device)
    models["bge_tokenizer"] = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    models["bge_model"] = AutoModel.from_pretrained("BAAI/bge-m3").to(device)
    if USE_SIMALIGN:
        models["aligner"] = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")  # CPU
    models["device"] = device
    print(f"[GPU SERVER] Models loaded on {device}")
    return models

def get_safe_max_len(model, tokenizer):
    if hasattr(model.config, "max_position_embeddings"):
        max_pos = int(model.config.max_position_embeddings)
    else:
        max_pos = int(getattr(tokenizer, "model_max_length", 512))
    return max(2, max_pos - 2)

def batched_model_encode_with_models(model, tokenizer, texts, device, batch_size=32, use_pooler=True, pooling_fn=None):
    if pooling_fn is None:
        pooling_fn = mean_pooling
    safe_max_len = get_safe_max_len(model, tokenizer)
    all_vecs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=safe_max_len, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            if use_pooler and hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled = outputs.pooler_output
            else:
                pooled = pooling_fn(outputs, encoded['attention_mask'])
            all_vecs.append(pooled.cpu())
    if all_vecs:
        return torch.cat(all_vecs, dim=0).numpy()
    else:
        return np.zeros((len(texts), model.config.hidden_size))

def gpu_get_embeddings(sentences, lang, models, batch_size=DEFAULT_BATCH_SIZE):
    """
    Embedding logic executed on GPU server using loaded 'models' dict.
    """
    n = len(sentences)
    if n == 0:
        return np.zeros((0, 0))

    device = models.get("device", MAIN_DEVICE)

    # SONAR
    if USE_SONAR:
        lang_tag = sonar_language_codes.get(lang.lower(), 'eng_Latn')
        emb_sonar_raw = models["text_embedder"].predict(sentences, source_lang=lang_tag)
        sonars = [e.detach().cpu().numpy() if torch.is_tensor(e) else np.array(e) for e in emb_sonar_raw]
        emb_sonar = np.vstack(sonars)
    else:
        emb_sonar = np.zeros((n, 1024))

    # LaBSE
    labse_raw = models["labse"].encode(
        sentences, convert_to_numpy=True,
        normalize_embeddings=False, batch_size=batch_size
    )
    emb_labse = np.array(labse_raw)

    # IndicBERT
    emb_indic = batched_model_encode_with_models(
        models["indic_model"], models["indic_tokenizer"],
        sentences, device, batch_size=batch_size,
        use_pooler=True, pooling_fn=mean_pooling
    )

    # GTE
    emb_gte = batched_model_encode_with_models(
        models["gte_model"], models["gte_tokenizer"],
        sentences, device, batch_size=batch_size,
        use_pooler=False, pooling_fn=mean_pooling
    )

    # BGE
    emb_bge = batched_model_encode_with_models(
        models["bge_model"], models["bge_tokenizer"],
        sentences, device, batch_size=batch_size,
        use_pooler=False, pooling_fn=mean_pooling
    )

    final = np.concatenate([emb_sonar, emb_labse, emb_indic, emb_gte, emb_bge], axis=1)
    norms = np.linalg.norm(final, axis=1, keepdims=True) + 1e-9
    final = final / norms

    # Explicit cleanup
    del emb_sonar_raw, sonars, labse_raw, emb_labse, emb_indic, emb_gte, emb_bge
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        pass

    return final

def gpu_server_loop(gpu_id, request_queue, response_store, stop_flag, batch_size=DEFAULT_BATCH_SIZE):
    """
    Process that runs on a specific GPU, loads models there and services embedding requests.
    """
    models = load_models_on_device(gpu_id)
    while not stop_flag.is_set():
        try:
            item = request_queue.get(timeout=0.5)
        except Exception:
            continue
        if item == "__STOP__":
            break
        try:
            req_id = item["req_id"]
            sentences = item["sentences"]
            lang = item["lang"]

            emb = gpu_get_embeddings(sentences, lang, models, batch_size=batch_size)

            response_store[req_id] = emb

            # Extra safety: free memory per request
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                pass

        except Exception as e:
            response_store[req_id] = None
            print(f"[GPU SERVER {gpu_id}] Error processing request {req_id}: {e}")

# -------------------------
# Proxy for embedding calls when in parallel mode
# -------------------------
def embed_via_gpu(sentences, lang, request_queues, response_store, timeout=120.0):
    """
    Round-robin across request_queues (one per GPU server).
    Uses a global semaphore to ensure we do NOT overload GPU with
    concurrent embedding jobs from multiple workers → prevents OOM.
    """
    with GPU_SEMAPHORE:
        req_id = str(uuid.uuid4())
        item = {"req_id": req_id, "sentences": sentences, "lang": lang}

        chosen_q = request_queues[hash(req_id) % len(request_queues)]
        chosen_q.put(item)

        t0 = time.time()
        while True:
            if req_id in response_store:
                res = response_store.pop(req_id)
                # Post-embedding GPU cleanup (extra safety)
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except:
                    pass
                return res

            if time.time() - t0 > timeout:
                raise TimeoutError(f"Embedding request {req_id} timed out")

            time.sleep(0.01)

# --------------------------------------------------------------------------------
# Sequential get_embeddings (original behavior)
# --------------------------------------------------------------------------------
def get_embeddings_sequential(sentences, lang, batch_size=64):
    n = len(sentences)
    if n == 0:
        return np.zeros((0, 0))

    if USE_SONAR:
        lang_tag = sonar_language_codes.get(lang.lower(), 'eng_Latn')
        emb_sonar_raw = text_embedder_seq.predict(sentences, source_lang=lang_tag)
        sonars = [e.detach().cpu().numpy() if torch.is_tensor(e) else np.array(e) for e in emb_sonar_raw]
        emb_sonar = np.vstack(sonars)
    else:
        emb_sonar = np.zeros((n, 1024))

    labse_raw = labse_model_seq.encode(
        sentences, convert_to_numpy=True,
        normalize_embeddings=False, batch_size=batch_size
    )
    emb_labse = np.array(labse_raw)

    emb_indic = batched_model_encode_with_models(
        indic_model_seq, indic_tokenizer_seq, sentences,
        MAIN_DEVICE, batch_size=batch_size,
        use_pooler=True, pooling_fn=mean_pooling
    )
    emb_gte = batched_model_encode_with_models(
        gte_model_seq, gte_tokenizer_seq, sentences,
        MAIN_DEVICE, batch_size=batch_size,
        use_pooler=False, pooling_fn=mean_pooling
    )
    emb_bge = batched_model_encode_with_models(
        bge_model_seq, bge_tokenizer_seq, sentences,
        MAIN_DEVICE, batch_size=batch_size,
        use_pooler=False, pooling_fn=mean_pooling
    )

    final = np.concatenate([emb_sonar, emb_labse, emb_indic, emb_gte, emb_bge], axis=1)
    norms = np.linalg.norm(final, axis=1, keepdims=True) + 1e-9
    final = final / norms

    del emb_sonar_raw, sonars, labse_raw, emb_labse, emb_indic, emb_gte, emb_bge
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except:
        pass

    return final

# -------------------------
# similarity & alignment (unchanged)
# -------------------------
def cosine_sim_matrix(A, B):
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]))
    a = torch.from_numpy(A).to(MAIN_DEVICE)
    b = torch.from_numpy(B).to(MAIN_DEVICE)
    sims = torch.matmul(a, b.t())
    return sims.cpu().numpy()

def align_sentences_topk(sentences1, sentences2, lang1, lang2='English',
                         threshold=0.75, top_k=3, max_usage=3, embed_fn=None):
    if embed_fn is None:
        embed_fn = get_embeddings_sequential

    emb1 = embed_fn(sentences1, lang1)
    emb2 = embed_fn(sentences2, lang2)

    aligned_pairs = []
    target_usage = defaultdict(int)

    sims = cosine_sim_matrix(emb1, emb2)

    n1, n2 = sims.shape
    for i in tqdm(range(n1), desc=f"Aligning {lang1}→{lang2}"):
        row = sims[i]
        top_indices = np.argsort(-row)[:min(n2, top_k * 10)]
        matches_added = 0
        for j in top_indices:
            sim = float(row[j])
            if sim < threshold:
                break
            if target_usage[j] < max_usage:
                aligned_pairs.append((sentences1[i], sentences2[j], sim))
                target_usage[j] += 1
                matches_added += 1
            if matches_added >= top_k:
                break
    return aligned_pairs

def backward_align(sentences1, sentences2, lang1, lang2='English',
                   threshold=0.75, top_k=3, max_usage=3, embed_fn=None):
    aligned_pairs = align_sentences_topk(
        sentences2, sentences1, lang2, lang1,
        threshold, top_k, max_usage, embed_fn=embed_fn
    )
    return [(src, tgt, sim) for src, tgt, sim in aligned_pairs]

def sliding_window_align(src_text, tgt_sentences, lang1, lang2,
                         min_window=20, max_window=300, step=10,
                         threshold=0.75, embed_fn=None):
    if embed_fn is None:
        embed_fn = get_embeddings_sequential
    src_text = normalize_text_new(src_text, lang1).replace('\n', ' ')
    aligned = []
    src_len = len(src_text)
    start_idx = 0

    for tgt in tqdm(tgt_sentences, desc=f"{lang1} → {lang2} char aligning"):
        best_sim = -1.0
        best_slice = ""
        tgt_emb = embed_fn([tgt], lang2)[0:1]
        tgt_len = len(tgt)
        dynamic_window = int(np.clip(tgt_len * 1.5, min_window, max_window))

        window_sizes = list(range(
            max(min_window, dynamic_window // 2),
            min(max_window, dynamic_window * 2),
            step
        ))
        window_texts = []
        slices = []
        for w in window_sizes:
            end_idx = min(start_idx + w, src_len)
            src_slice = src_text[start_idx:end_idx]
            if len(src_slice.strip()) < 5:
                continue
            window_texts.append(src_slice)
            slices.append(src_slice)

        if not window_texts:
            continue

        windows_emb = embed_fn(window_texts, lang1)
        sims = cosine_sim_matrix(windows_emb, tgt_emb).flatten()
        idx = int(np.argmax(sims))
        if sims[idx] > best_sim:
            best_sim = float(sims[idx])
            best_slice = slices[idx]

        if best_sim > threshold:
            aligned.append((best_slice, tgt, best_sim))
            start_idx += len(best_slice)
        else:
            start_idx += max(1, dynamic_window // 3)
        if start_idx >= src_len:
            break
    return aligned

def sliding_window_align_words(src_text, tgt_sentences, lang1, lang2,
                               min_window=5, max_window=40, step=1,
                               threshold=0.75, debug=False, embed_fn=None):
    if embed_fn is None:
        embed_fn = get_embeddings_sequential
    src_text = normalize_text_new(src_text, lang1).replace('\n', ' ')
    src_words = src_text.split()
    total_words = len(src_words)
    aligned = []
    start_idx = 0

    for tgt in tqdm(tgt_sentences, desc=f"{lang1} → {lang2} word aligning"):
        best_sim = -1.0
        best_slice = ""
        tgt_emb = embed_fn([tgt], lang2)[0:1]
        tgt_word_count = len(tgt.split())
        dynamic_window = int(np.clip(tgt_word_count * 1.5, min_window, max_window))

        window_texts = []
        slices = []
        for w in range(
            max(min_window, dynamic_window // 2),
            min(max_window, dynamic_window * 2),
            step
        ):
            end_idx = min(start_idx + w, total_words)
            src_slice_words = src_words[start_idx:end_idx]
            src_slice = " ".join(src_slice_words)
            if len(src_slice.strip()) < 5:
                continue
            window_texts.append(src_slice)
            slices.append(src_slice)

        if not window_texts:
            start_idx += max(1, dynamic_window // 3)
            continue

        windows_emb = embed_fn(window_texts, lang1)
        sims = cosine_sim_matrix(windows_emb, tgt_emb).flatten()
        idx = int(np.argmax(sims))
        if sims[idx] > best_sim:
            best_sim = float(sims[idx])
            best_slice = slices[idx]

        if best_sim > threshold:
            aligned.append((best_slice, tgt, best_sim))
            consumed_words = len(best_slice.split())
            start_idx += consumed_words
        else:
            start_idx += max(1, dynamic_window // 3)

        if start_idx >= total_words:
            break
    return aligned

def get_word_alignment(src, tgt):
    if USE_SIMALIGN:
        aligns = aligner_seq.get_word_aligns(src, tgt)['inter']
        score = len(aligns) / max(len(src.split()), len(tgt.split())) if max(len(src.split()), len(tgt.split())) > 0 else 0
        return aligns, score
    return [], 0

def save_list_column_to_lines(df, column, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in df[column].dropna().drop_duplicates():
            if isinstance(row, list):
                for item in row:
                    f.write(f"{item}\n")
            else:
                f.write(f"{row}\n")

def cosine_sim(v1, v2):
    v1 = np.asarray(v1)
    v2 = np.asarray(v2).flatten()
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))

def merge_forward_backward(fwd, bwd):
    merged_src = set(src for src, tgt, sim in fwd)
    extra = [(src, tgt, sim) for src, tgt, sim in bwd if src not in merged_src]
    return fwd + extra

# -------------------------
# Episode-level processing
# -------------------------
def process_episode(episode, args, embed_fn=None):
    text_dir = args.text_dir
    output_dir = args.output_dir
    threshold = args.threshold

    languages = sorted([l for l in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, l))])
    texts = {}
    split_sentences_dict = {}

    for lang in languages:
        file_path = os.path.join(args.text_dir, lang, episode, f"{episode}.txt")
        if os.path.exists(file_path):
            texts[lang] = read_file(file_path)
        else:
            tqdm.write(f"⚠️ Missing file for {lang}/{episode}: {file_path}")

    for lang, raw_text in texts.items():
        try:
            lang_code = language_codes.get(lang, None)
            sents = split_sentences(raw_text, lang_code=lang_code, min_len=5, use_buffer=True, normalize=False)
            sents = filter_language_mismatch(sents, lang) if sents else []
            split_sentences_dict[lang] = sents
            tqdm.write(f"  ▸ {lang}: {len(sents)} sentences")
        except Exception as e:
            tqdm.write(f"⚠️ Failed splitting for {lang}: {e}")
            split_sentences_dict[lang] = [raw_text] if raw_text.strip() else []

    output = os.path.join(args.output_dir, episode)
    os.makedirs(output, exist_ok=True)

    parallel_corpus = []
    if "English" not in split_sentences_dict or not split_sentences_dict["English"]:
        tqdm.write(f"⚠️ No English sentences found for episode {episode} — skipping alignment.")
        return None

    english_sents = split_sentences_dict["English"]

    for lang in languages:
        if lang == "English":
            continue
        if lang not in split_sentences_dict or not split_sentences_dict[lang]:
            tqdm.write(f"⚠️ No sentences for {lang} — skipping.")
            continue

        fwd = align_sentences_topk(
            split_sentences_dict[lang], english_sents, lang, lang2="English",
            threshold=args.threshold, top_k=args.top_k, max_usage=args.max_usage,
            embed_fn=embed_fn
        )
        bwd = backward_align(
            english_sents, split_sentences_dict[lang],
            "English", lang, threshold=args.threshold,
            top_k=args.top_k, max_usage=args.max_usage, embed_fn=embed_fn
        )
        merged = merge_forward_backward(fwd, bwd)

        sliding_char = sliding_window_align(
            texts[lang], english_sents, lang, "English",
            min_window=args.min_window, max_window=args.max_window,
            step=args.step, threshold=args.threshold, embed_fn=embed_fn
        )
        sliding_words = sliding_window_align_words(
            texts[lang], english_sents, lang, "English",
            min_window=args.min_window, max_window=100,
            step=1, threshold=args.threshold, debug=False, embed_fn=embed_fn
        )

        combined = merged + sliding_char + sliding_words

        src_lang_code = language_codes.get(lang, None)
        final_pairs = []
        for src, tgt, sim in combined:
            try:
                normalized_src = normalize_text_new(src, src_lang_code) if src_lang_code else normalize_text_new(src, "")
                final_pairs.append((normalized_src, tgt, sim))
            except Exception as e:
                tqdm.write(f"⚠️ Normalization failed for {lang} sentence: {e}")
                final_pairs.append((src, tgt, sim))

        if final_pairs:
            parallel_corpus.append((lang, final_pairs))
            tqdm.write(f"  ✓ {lang}: {len(final_pairs)} aligned pairs")
        else:
            tqdm.write(f"  ⚠️ {lang}: no aligned pairs produced")

    if not parallel_corpus:
        tqdm.write(f"⚠️ No alignments produced for episode {episode} — nothing to save.")
        return None

    merged_df = None
    for lang, pairs in parallel_corpus:
        df = pd.DataFrame(pairs, columns=[lang, "English", "similarity"]).drop_duplicates().dropna()
        df.to_excel(os.path.join(output, f"{lang}_aligned.xlsx"), index=False)
        df_grouped = df.groupby("English")[lang].apply(lambda x: list(sorted(set(x)))).reset_index()
        merged_df = df_grouped if merged_df is None else pd.merge(merged_df, df_grouped, on="English", how="outer")

    if merged_df is None or merged_df.empty:
        tqdm.write(f"⚠️ After processing, merged_df is empty for episode {episode}. Skipping final save.")
        return None

    combined_excel = os.path.join(output, "all_aligned.xlsx")
    merged_df.to_excel(combined_excel, index=False)
    tqdm.write(f"✅ Saved combined Excel: {combined_excel}")

    merged_df.to_json(os.path.join(output, "all_aligned.json"), orient="records", force_ascii=False, indent=0)

    for col in merged_df.columns:
        out_dir = os.path.join(args.text_dir, col)
        os.makedirs(out_dir, exist_ok=True)
        save_list_column_to_lines(merged_df, col, os.path.join(out_dir, episode, f"{episode}_en_aligned.txt"))

    merged_df = merged_df.dropna()
    if not merged_df.empty:
        combined_clean_excel = os.path.join(output, "all_aligned_without_null.xlsx")
        merged_df.to_excel(combined_clean_excel, index=False)
        tqdm.write(f"✅ Saved cleaned Excel: {combined_clean_excel}")
        merged_df.to_json(os.path.join(output, "all_aligned_without_null.json"), orient="records", force_ascii=False, indent=0)
        for col in merged_df.columns:
            save_list_column_to_lines(merged_df, col, os.path.join(args.text_dir, col, episode, f"{episode}_all_aligned.txt"))
        tqdm.write(f"✅ Saved final cleaned aligned file for episode {episode}")
    else:
        tqdm.write(f"⚠️ No non-null rows after cleaning for episode {episode}.")
    return merged_df

# -------------------------
# Orchestration: parallel and sequential entrypoints
# -------------------------
def run_sequential_mode(args):
    text_dir = args.text_dir
    languages = sorted([l for l in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, l))])
    if not languages:
        raise ValueError("No language folders found.")
    first_lang_dir = os.path.join(args.text_dir, languages[0])
    '''episode_folders = sorted(
        os.listdir(first_lang_dir),
        key=lambda x: int(re.search(r'\d+', x).group())
    ) if os.listdir(first_lang_dir) else []
    #episode_folders = episode_folders[98:]'''
    
    episode_folders = ['MKB_77_May_2021','MKB_98_February_2023','MKB_78_June_2021']

    for episode in tqdm(episode_folders, desc="Processing episodes (sequential)"):
        process_episode(episode, args, embed_fn=get_embeddings_sequential)

def run_parallel_mode(args):
    n_gpus = max(1, min(
        args.n_gpus,
        torch.cuda.device_count() if torch.cuda.is_available() else 0 or 1
    ))
    if n_gpus <= 0:
        print("No GPUs detected; falling back to sequential mode.")
        return run_sequential_mode(args)

    manager = mp.Manager()
    request_queues = []
    response_store = manager.dict()

    for gpu_id in range(n_gpus):
        q = manager.Queue()
        request_queues.append(q)

    gpu_processes = []
    stop_events = []
    for gpu_id in range(n_gpus):
        stop_flag = mp.Event()
        p = mp.Process(
            target=gpu_server_loop,
            args=(gpu_id, request_queues[gpu_id], response_store, stop_flag, DEFAULT_BATCH_SIZE),
            daemon=True
        )
        p.start()
        gpu_processes.append(p)
        stop_events.append(stop_flag)
        print(f"[LAUNCHER] Started GPU server on device {gpu_id}")

    text_dir = args.text_dir
    languages = sorted([l for l in os.listdir(text_dir) if os.path.isdir(os.path.join(text_dir, l))])
    if not languages:
        raise ValueError("No language folders found.")
    first_lang_dir = os.path.join(args.text_dir, languages[0])
    episode_folders = sorted(
        os.listdir(first_lang_dir),
        key=lambda x: int(re.search(r'\d+', x).group())
    ) if os.listdir(first_lang_dir) else []
    '''episode_folders = episode_folders[98:]
    episode_folders.extend([
        'MKB_76_April_2021','MKB_74_February_2021','MKB_94_October_2022',
        'MKB_90_June_2022','MKB_57_September_2019','MKB_77_May_2021','MKB_78_June_2021'
    ])
    episode_folders = sorted(
        episode_folders,
        key=lambda x: int(re.search(r'\d+', x).group())
    )'''

    def worker_target(episode):
        try:
            def worker_embed_fn(sentences, lang):
                return embed_via_gpu(sentences, lang, request_queues, response_store)
            return process_episode(episode, args, embed_fn=worker_embed_fn)
        except Exception as e:
            print(f"[worker] error processing {episode}: {e}")
            return None

    from multiprocessing.pool import ThreadPool
    total_workers = args.workers_per_gpu * n_gpus
    pool = ThreadPool(processes=total_workers)
    async_results = []

    for episode in episode_folders:
        async_results.append(pool.apply_async(worker_target, (episode,)))

    pool.close()

    results = []
    for ar in async_results:
        try:
            r = ar.get()
            results.append(r)
        except Exception as e:
            print("Worker exception:", e)

    pool.join()

    for q in request_queues:
        q.put("__STOP__")
    for p in gpu_processes:
        p.terminate()
        p.join()

    return results

def run_pipeline(args):
    if args.run_mode == "sequential":
        run_sequential_mode(args)
    else:
        run_parallel_mode(args)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["sliding","topk","all"], default="all")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--min_window", type=int, default=1)
    parser.add_argument("--max_window", type=int, default=350)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--dynamic_scale", type=float, default=1.5)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--max_usage", type=int, default=1)
    parser.add_argument("--use_simalign", action="store_true")
    parser.add_argument("--run_mode", choices=["sequential","parallel"], default="sequential")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    args = parser.parse_args()
    run_pipeline(args)

