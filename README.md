Here's a complete, professional `README.md` file tailored for your repository that includes **both the transcription** (`transcribe.py`) and **alignment** (`align.py`) functionalities. It presents each module clearly, documents all supported languages, and explains how users can run each script.

---

```markdown
# 🗣️ Multilingual Transcription & Sentence Alignment Toolkit

This repository provides tools to:
1. **Transcribe multilingual audio files** using [Meta AI's MMS model](https://huggingface.co/facebook/mms-1b-all), and
2. **Align transcriptions** with translations using embedding-based sentence alignment (e.g., Sonar or LaBSE + SimAlign).

---

## 📌 Features

- 🔊 Transcribe `.wav` audio files across **36 Indian languages**
- 🧠 Leverage **Facebook's MMS (Massively Multilingual Speech)** model
- ✂️ Chunked long-audio handling with silence-skipping
- 🌐 Language-specific tokenizers and adapter loading
- 🌍 Align transcripts with translations using **SimAlign** and **Sonar/LabSE**
- 📝 Output sentence-aligned `.csv` files for downstream NLP tasks

---

## 📂 Directory Structure

```

project-root/
├── transcribe.py            # Transcribes .wav files to text
├── align.py                 # Aligns transcribed text with translations
├── requirements.txt
├── README.md
└── /data/
├── Hindi/
│   ├── MKB\_101/
│   │   ├── MKB\_101.wav
│   │   └── MKB\_101.txt
├── English/
│   ├── MKB\_101/
│   │   └── MKB\_101.txt
└── ...

````

---

## 🧱 Requirements

Install required libraries:

```bash
pip install -r requirements.txt
````

### `requirements.txt` includes:

```text
torch
torchaudio
transformers
pandas
tqdm
scipy
simalign
```

---

## 1️⃣ Audio Transcription (`transcribe.py`)

### ▶️ Description

Transcribes `.wav` audio files in various Indian languages using the **MMS model** and saves them as `.txt` files.

### 🧾 Usage

```bash
python transcribe.py --root_dir /path/to/data
```

* `--root_dir`: Root directory containing subfolders by language (e.g., `Hindi`, `Assamese`).

### ⚙️ Features

* Uses HuggingFace `Wav2Vec2ForCTC` + `AutoProcessor`
* Automatically maps each language to the appropriate tokenizer and adapter
* Chunks long audio into 10-second segments
* Skips low-quality audio chunks

---

## 2️⃣ Sentence Alignment (`align.py`)

### ▶️ Description

Aligns transcriptions (e.g., in Hindi) with their English translations on a sentence level using embedding similarity.

### 🔧 Options

```bash
python align.py \
  --src_lang Hindi \
  --tgt_lang English \
  --root_dir /path/to/data \
  --output_dir /path/to/output \
  --embed_model sonar \
  --aligner simalign
```

### 📥 Required Inputs

Each language should have `.txt` files for corresponding episodes, e.g.:

```
/Hindi/MKB_101/MKB_101.txt
/English/MKB_101/MKB_101.txt
```

### 📤 Output

For each episode, the output CSV will contain:

| src               | tgt                 | sim\_score |
| ----------------- | ------------------- | ---------- |
| \[Hindi sentence] | \[English sentence] | 0.87       |

---

## 🌐 Supported Languages

| Language     | MMS Code            |
| ------------ | ------------------- |
| Assamese     | `asm`               |
| Bengali      | `ben`               |
| Bodo         | `asm`               |
| Chattisgarhi | `hin`               |
| Dogri        | `hin`               |
| English      | `eng`               |
| Garo         | `bod`               |
| Galo         | `bod`               |
| Gujarati     | `guj`               |
| Hindi        | `hin`               |
| Jaintia      | `bod`               |
| Kannada      | `kan`               |
| Kashmiri     | `urd-script_arabic` |
| Khasi        | `bod`               |
| Kokborok     | `ben`               |
| Konkani      | `hin`               |
| Ladakhi      | `bod`               |
| Lepcha       | `bod`               |
| Maithili     | `hin`               |
| Malayalam    | `mal`               |
| Manipuri     | `ben`               |
| Marathi      | `mar`               |
| Mizo         | `bod`               |
| Nepali       | `hin`               |
| Odia         | `ory`               |
| Punjabi      | `pan`               |
| Purgi        | `urd-script_arabic` |
| Sanskrit     | `hin`               |
| Santhali     | `ben`               |
| Sargujia     | `hin`               |
| Sikkimese    | `bod`               |
| Sindhi       | `hin`               |
| Tamil        | `tam`               |
| Telugu       | `tel`               |
| Urdu         | `urd-script_arabic` |

---

## ✏️ Customization

* **Change chunk size** (default is 10s): Edit `chunk_duration_sec` in `transcribe_audio(...)`
* **Filter languages**: Adjust the language list or `lang_map` in the script
* **Choose alignment backend**: Use `--embed_model sonar` or `labse` in `align.py`

---

## 📎 License

This project uses:

* [Meta MMS models](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
* [HuggingFace Transformers](https://huggingface.co/facebook/mms-1b-all)
* [SimAlign](https://github.com/cisnlp/SimAlign)

Please refer to each library's license for usage terms.

---

## 🙏 Acknowledgements

* Meta AI for the MMS models
* HuggingFace for providing model hosting and APIs
* SimAlign for sentence alignment algorithms

---

## 📬 Contact & Contributions

If you find a bug or want to contribute:

* Open an issue or pull request
* For questions, feel free to reach out via GitHub

---

```

---

### ✅ Next Steps

Would you like me to:
- Export this README and supporting files into a `.zip`?
- Add examples of `.txt` input/output?
- Help with pushing this to GitHub?

Let me know how you'd like to proceed.
```
