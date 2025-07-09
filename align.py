# Enhanced version with LaBSE, SimAlign integration, and main function

import argparse
import os
import re
import torch
import pandas as pd
import numpy as np
from numpy import ndarray
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from simalign import SentenceAligner
from tqdm import tqdm

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tqdm.write(f"🔹 Using device: {DEVICE}")

# Configuration
USE_SONAR = True  # Toggle to use Sonar or LaBSE
USE_SIMALIGN = True  # Toggle to use SimAlign

# Language code dictionaries (trimmed)
language_codes = {
    "English": "eng", "Bodo": "asm", "Chattisgarhi": "hin", "Dogri": "hin",
    "Garo": "bod", "Galo": "bod", "Jaintia": "bod", "Kashmiri": "urd-script_arabic",
    "Khasi": "bod", "Kokborok": "ben", "Konkani": "hin", "Ladakhi": "bod",
    "Lepcha": "bod", "Maithili": "hin", "Mizo": "bod", "Nepali": "hin",
    "Purgi": "urd-script_arabic", "Sanskrit": "hin", "Santhali": "ben",
    "Sargujia": "hin", "Sikkimese": "bod", "Sindhi": "hin",
    "Assamese": "asm", "Bengali": "ben", "Gujarathi": "guj",
    "Hindi": "hin", "Kannada": "kan", "Malayalam": "mal", "Manipuri": "ben",
    "Marathi": "mar", "Odia": "ory", "Punjabi": "pan", "Tamil": "tam",
    "Telugu": "tel", "Urdu": "urd-script_arabic"
}

sonar_language_codes = {
    "English": "eng_Latn", "Bodo": "asm_Beng", "Chattisgarhi": "hin_Deva", "Dogri": "hin_Deva",
    "Garo": "bod_Tibt", "Galo": "bod_Tibt", "Jaintia": "bod_Tibt", "Kashmiri": "urd_Arab",
    "Khasi": "bod_Tibt", "Kokborok": "ben_Beng", "Konkani": "hin_Deva", "Ladakhi": "bod_Tibt",
    "Lepcha": "bod_Tibt", "Maithili": "hin_Deva", "Mizo": "bod_Tibt", "Nepali": "hin_Deva",
    "Purgi": "urd_Arab", "Sanskrit": "hin_Deva", "Santhali": "ben_Beng",
    "Sargujia": "hin_Deva", "Sikkimese": "bod_Tibt", "Sindhi": "hin_Deva",
    "Assamese": "asm_Beng", "Bengali": "ben_Beng", "Gujarathi": "guj_Gujr",
    "Hindi": "hin_Deva", "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Manipuri": "ben_Beng",
    "Marathi": "mar_Deva", "Odia": "ory_Orya", "Punjabi": "pan_Guru", "Tamil": "tam_Taml",
    "Telugu": "tel_Telu", "Urdu": "urd_Arab"
}

# Load models
if USE_SONAR:
    tqdm.write("🔹 Loading Sonar model...")
    text_embedder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=DEVICE
    )
else:
    tqdm.write("🔹 Loading LaBSE model...")
    model = SentenceTransformer('sentence-transformers/LaBSE', device=str(DEVICE))

if USE_SIMALIGN:
    tqdm.write("🔹 Loading SimAlign model...")
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")


# Utility Functions

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def split_sentences(text,min_len=10):
    sentence_enders = r'[।?؟॥።։܀۔⁇⁈⁉‼⸮⸼…—\~\n\r\t\.|]'
    pattern = f'({sentence_enders}+)\\s*'
    sentences = re.split(pattern, text.strip())
    combined = [
        (sentences[i] + sentences[i + 1]).strip()
        for i in range(0, len(sentences) - 1, 2)
    ]
    if len(sentences) % 2 == 1:
        combined.append(sentences[-1].strip())
    #return [s for s in combined if s]
    result = []
    buffer = ""
    for sentence in combined:
        if len(sentence) <= min_len or re.fullmatch(r'[\W_]+', sentence):
            buffer += " " + sentence
        else:
            if buffer:
                # Merge buffer to the last item if exists
                if result:
                    result[-1] += buffer
                else:
                    result.append(buffer.strip())
                buffer = ""
            result.append(sentence.strip())

    # Merge leftover buffer
    if buffer:
        if result:
            result[-1] += buffer
        else:
            result.append(buffer.strip())

    return [s.strip() for s in result if s.strip()]


def get_embeddings(sentences, lang):
    if USE_SONAR:
        lang_tag = sonar_language_codes.get(lang, "eng_Latn")
        embeddings = text_embedder.predict(sentences, source_lang=lang_tag)
        return [torch.tensor(emb).to(DEVICE).cpu().numpy() for emb in embeddings]
    else:
        return model.encode(sentences, device=str(DEVICE))

def align_sentences(sentences1, sentences2, lang1, lang2="English"):
    tqdm.write("🔹 Generating embeddings...")
    embeddings1 = get_embeddings(sentences1, lang1)
    embeddings2 = get_embeddings(sentences2, lang2)

    tqdm.write("🔹 Computing similarity scores with uniqueness constraint...")

    used_indices = set()
    aligned_pairs = []

    for i, emb1 in tqdm(enumerate(embeddings1), total=len(embeddings1), desc=f"🔁 Aligning {lang1} → {lang2}"):
        if isinstance(emb1, np.ndarray) and emb1.ndim > 1:
            emb1 = emb1.flatten()

        best_score, best_match_idx = float("inf"), -1
        for j, emb2 in enumerate(embeddings2):
            if j in used_indices:
                continue  # Skip already aligned target sentence

            if isinstance(emb2, np.ndarray) and emb2.ndim > 1:
                emb2 = emb2.flatten()

            score = cosine(np.array(emb1), np.array(emb2))
            if score < best_score:
                best_score = score
                best_match_idx = j

        if best_match_idx != -1:
            used_indices.add(best_match_idx)
            aligned_pairs.append((sentences1[i], sentences2[best_match_idx]))
        else:
            aligned_pairs.append((sentences1[i], None))  # No match found
            
    for i, emb1 in tqdm(enumerate(embeddings1), total=len(embeddings1), desc=f"🔁 Aligning {lang1} -> {lang2}"):
        if isinstance(emb1, ndarray) and emb1.ndim > 1:
            emb1 = emb1.flatten()
        best_score, best_match = float("inf"), None
        for j, emb2 in enumerate(embeddings2):
            if isinstance(emb2, ndarray) and emb2.ndim > 1:
                emb2 = emb2.flatten()
            score = cosine(np.array(emb1), np.array(emb2))
            if score < best_score:
                best_score = score
                best_match = (sentences1[i], sentences2[j])
        aligned_pairs.append(best_match)
        
    for i, emb2 in tqdm(enumerate(embeddings2), total=len(embeddings2), desc=f"🔁 Aligning {lang1} -> {lang2}"):
        if isinstance(emb2, ndarray) and emb2.ndim > 1:
            emb2 = emb2.flatten()
        best_score, best_match = float("inf"), None
        for j, emb1 in enumerate(embeddings1):
            if isinstance(emb1, ndarray) and emb1.ndim > 1:
                emb1 = emb1.flatten()
            score = cosine(np.array(emb2), np.array(emb1))
            if score < best_score:
                best_score = score
                best_match = (sentences1[j], sentences2[i])
        aligned_pairs.append(best_match)

    return aligned_pairs



def get_word_alignment(src_sent, tgt_sent):
    if USE_SIMALIGN:
        result = aligner.get_word_aligns(src_sent, tgt_sent)
        inter_align = result['inter']
        word_match_score = len(inter_align) / max(len(src_sent.split()), len(tgt_sent.split()))
        return inter_align, word_match_score
    return [], 0


def align_stanza_sentences(lang1_sentences, english_sentences, lang1, lang2="English"):
    tqdm.write(f"🔹 Aligning {len(lang1_sentences)} {lang1} sentences with {len(english_sentences)} English sentences...")
    return align_sentences(lang1_sentences, english_sentences, lang1, lang2)


def extract_episode_number(folder_name):
    match = re.search(r'MKB_(\d+)', folder_name)
    return int(match.group(1)) if match else float('inf')

def save_list_column_to_lines(df, column, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for row in df[column].dropna():
            if isinstance(row, list):
                for item in row:
                    f.write(f"{item}\n")
            else:
                f.write(f"{row}\n")

def main(args):
    languages = sorted(os.listdir(args.text_dir))
    episodes = sorted(os.listdir(os.path.join(args.text_dir, languages[0])), key=extract_episode_number, reverse=True)
    
    if args.text_dir == "/DATA/nfsshare/Adarsh/Multilingual/Maan_ki_Baat_all_36_languages":
        key=episodes.index("MKB_49_October_2018")
        episodes=episodes[key:]
    for episode in tqdm(episodes, desc="Processing episodes across all languages"):
        texts, split_sentences_dict = {}, {}

        for lang in languages:
            file_path = os.path.join(args.text_dir, lang, episode, f"{episode}.txt")
            if os.path.exists(file_path):
                texts[lang] = read_file(file_path)
            else:
                tqdm.write(f"⚠️ Warning: {file_path} not found.")

        output = os.path.join(args.output_dir, episode)
        os.makedirs(output, exist_ok=True)

        for lang, text in texts.items():
            tqdm.write(f"\n🔹 Processing {lang}...")
            sentences = split_sentences(text,min_len=10)
            tqdm.write(f"{lang} - {len(sentences)} sentences extracted.")
            split_sentences_dict[lang] = sentences

            pd.DataFrame(sentences).to_csv(
                os.path.join(args.text_dir, lang, episode, f"{episode}_split_sentences.txt"),
                header=False, index=False, sep="\n"
            )

        parallel_corpus, merged_df = [], None
        for lang in split_sentences_dict:
            if lang == "English":
                continue
            aligned = align_stanza_sentences(split_sentences_dict[lang], split_sentences_dict["English"], lang)
            parallel_corpus.append((lang, aligned))

        for lang, pairs in parallel_corpus:
            df = pd.DataFrame(pairs, columns=[lang, "English"]) 
            df=df.drop_duplicates()
            df=df.dropna()         
            df.to_excel(os.path.join(output, f"{lang}_aligned.xlsx"), index=False)
            #df=df.drop_duplicates()
            df = df.groupby("English")[lang].apply(lambda x: list(sorted(set(x)))).reset_index()
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on="English", how="outer")
            tqdm.write(f"✅ Saved aligned file for {lang}")

        merged_df = merged_df[["English"] + [col for col in merged_df.columns if col != "English"]]

        # Save as Parquet instead of Excel due to size limits
        parquet_file = os.path.join(output, "all_aligned.xlsx")
        merged_df.to_excel(parquet_file, index=False)
        tqdm.write(f"✅ Saved as Excel: {parquet_file}")
        merged_df.to_json(os.path.join(output, "all_aligned.json"), orient="records", force_ascii=False, indent=0)

        for col in merged_df.columns:
            save_list_column_to_lines(merged_df,col,os.path.join(args.text_dir, col, episode, f"{episode}_en_aligned.txt"))
            
        merged_df=merged_df.dropna()
        parquet_file = os.path.join(output, "all_aligned_without_null.xlsx")
        merged_df.to_excel(parquet_file, index=False)
        tqdm.write(f"✅ Saved as Excel: {parquet_file}")
        merged_df.to_json(os.path.join(output, "all_aligned_without_null.json"), orient="records", force_ascii=False, indent=0)
        for col in merged_df.columns:
            save_list_column_to_lines(merged_df,col,os.path.join(args.text_dir, col, episode, f"{episode}_all_aligned.txt"))
        tqdm.write(f"✅ Saved final cleaned aligned file")
        #break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multilingual Sentence Alignment using Sonar or LaBSE + SimAlign")
    parser.add_argument('--text_dir', required=True, help="Directory containing multilingual text files.")
    parser.add_argument('--output_dir', required=True, help="Output directory for aligned files.")
    args = parser.parse_args()
    main(args)
