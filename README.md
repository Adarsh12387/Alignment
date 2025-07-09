# 🗣️ Multilingual Transcription and Alignment Using Facebook MMS

This repository provides Python scripts for:

- ✅ **Transcribing multilingual `.wav` audio files** using [Meta's MMS (Massively Multilingual Speech) model](https://huggingface.co/facebook/mms-1b-all)
- ✅ **Aligning transcribed audio** with sentence-level timestamps using [MMS Forced Aligner](https://github.com/facebookresearch/seamless_communication)

---
## 📁 Project Structure
```text
project-root/
├── transcribe.py # Transcribes multilingual audio using MMS
├── align.py # Aligns audio with sentence-level segments
├── requirements.txt # Python dependencies
└── README.md # Project documentation
## 🔤 Supported Languages
```
Here are the languages supported by your current `lang_map`:

| Language       | MMS Language Code      |
|----------------|------------------------|
| English        | eng                    |
| Assamese       | asm                    |
| Bengali        | ben                    |
| Gujarathi      | guj                    |
| Hindi          | hin                    |
| Kannada        | kan                    |
| Malayalam      | mal                    |
| Manipuri       | ben                    |
| Marathi        | mar                    |
| Odia           | ory                    |
| Punjabi        | pan                    |
| Tamil          | tam                    |
| Telugu         | tel                    |
| Urdu           | urd-script_arabic      |
| Bodo           | asm                    |
| Chattisgarhi   | hin                    |
| Dogri          | hin                    |
| Garo           | bod                    |
| Galo           | bod                    |
| Jaintia        | bod                    |
| Kashmiri       | urd-script_arabic      |
| Khasi          | bod                    |
| Kokborok       | ben                    |
| Konkani        | hin                    |
| Ladakhi        | bod                    |
| Lepcha         | bod                    |
| Maithili       | hin                    |
| Mizo           | bod                    |
| Nepali         | hin                    |
| Purgi          | urd-script_arabic      |
| Sanskrit       | hin                    |
| Santhali       | ben                    |
| Sargujia       | hin                    |
| Sikkimese      | bod                    |
| Sindhi         | hin                    |

---

## 🧰 Requirements

Install all Python dependencies:

```bash
pip install -r requirements.txt
```
📂 Input Directory Structure
Organize your audio files by language and session inside a root directory:
```text 
root_dir/
├── Hindi/
│ └── MKB_120_March_2025/
│     └── MKB_120_March_2025.wav
├── Assamese/
│ └── MKB_110_Jan_2024/
│   └── MKB_110_Jan_2024.wav
``` 
📝 Note: Subdirectories under each language can follow any naming convention, as long as they contain a .wav file.

