import os
import re
import logging
import argparse
import pandas as pd
import torchaudio
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2ForCTC
from seamless_communication.datasets.huggingface import SpeechTokenizer  # Corrected import

# Language code map for MMS-compatible language tags
lang_map = {
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

def setup_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Multilingual Audio Transcription using MMS")
    parser.add_argument("--root_dir", required=True, help="Root directory containing .wav files")
    return parser.parse_args()

def extract_episode_number(folder_name):
    match = re.search(r'MKB_(\d+)', folder_name)
    return int(match.group(1)) if match else float('inf')
    
def extract_seq_number(file_path):
    match = re.search(r'seq_(\d+)', file_path)
    return int(match.group(1)) if match else float('inf')

def transcribe_audio(processor, model, language, logger, audio_path: str, output_txt_path: str, chunk_duration_sec: int = 10, sampling_rate: int = 16000):
    """
    Transcribes a long audio file by splitting it into chunks and saving the full transcription.
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != sampling_rate:
        resampler = torchaudio.transforms.Resample(sr, sampling_rate)
        waveform = resampler(waveform)

    waveform = waveform[0]  # mono
    total_samples = waveform.shape[0]
    chunk_samples = chunk_duration_sec * sampling_rate

    processor.tokenizer.set_target_lang(language)
    model.load_adapter(language)

    total_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    transcriptions = []

    for i in tqdm(range(0, total_samples, chunk_samples), desc="🔊 Transcribing", unit="chunk"):
        chunk = waveform[i:i + chunk_samples]
        if chunk.shape[0] < 4000:
            print(f"⚠️ Skipping short chunk at {i / sampling_rate:.2f}s ({chunk.shape[0]} samples)")
            continue

        inputs = processor(chunk.numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 🛠 Fix: move inputs to model device

        input_values = inputs["input_values"]
        if input_values.shape[1] < 4000:
            print(f"⚠️ Skipping processed input: too short after tokenization ({input_values.shape[1]} samples)")
            continue

        with torch.no_grad():
            logits = model(**inputs).logits

        ids = torch.argmax(logits, dim=-1)[0]
        transcription = processor.decode(ids, skip_special_tokens=True)
        transcriptions.append(transcription)

    final_transcription = "\n".join(transcriptions)
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(final_transcription)

    print(f"\n✅ Transcription saved to: {output_txt_path}")
    
def main():
    args = parse_args()
    logger = setup_logger()

    root_dir = args.root_dir
    if not os.path.isdir(root_dir):
        logger.error(f"Provided root_dir is not a valid directory: {root_dir}")
        return

    languages_list = sorted(os.listdir(root_dir))
    logger.info("Loading MMS model...")

    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
   
    for language in tqdm(languages_list, desc="Processing languages"):
        if language in [
            "Assamese", "Bengali", "English", "Gujarathi", "Hindi", "Kannada", "Malayalam",
            "Manipuri", "Marathi", "Odia", "Punjabi", "Tamil", "Telugu", "Urdu"]:
            #,"Bodo", "Chattisgarhi", "Dogri", "Garo", "Galo", "Jaintia", "Kashmiri"
        #]:
            continue

        lang_path = os.path.join(root_dir, language)
        if not os.path.isdir(lang_path):
            continue
        if language not in lang_map:
            logger.warning(f"Language '{language}' not in lang_map. Skipping...")
            continue

        episode_folders = sorted(
            os.listdir(lang_path),
            key=extract_episode_number,
            reverse=True
        )

        """if language == "Khasi":
            try:
                ind = episode_folders.index("MKB_83_November_2021")
                episode_folders = episode_folders[ind:]
            except ValueError:
                logger.warning("MKB_83_November_2021 not found in Khasi folder list.")"""

        for episode in tqdm(episode_folders, desc=f"Processing episodes for {language}"):
            wav_dir = os.path.join(lang_path, episode)
            if not os.path.isdir(wav_dir):
                continue

            for file in os.listdir(wav_dir):
                if file.endswith(".wav"):
                    audio_path = os.path.join(wav_dir, file)
                    txt_path = audio_path.replace(".wav", ".txt")
                    transcribe_audio(
                        processor, model, lang_map[language], logger,
                        audio_path, txt_path, 10,sampling_rate=16000
                    )

if __name__ == "__main__":
    main()
