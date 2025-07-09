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
##📂 Input Directory Structure
Organize your audio files by language and session inside a root directory:
```text 
root_dir/
├── Hindi/
│ └── MKB_120_March_2025/
│     └── MKB_120_March_2025.wav
├── Assamese/
│ └──MKB_120_March_2025/
│   └── MKB_120_March_2025.wav
``` 
📝 Note: Subdirectories under each language can follow any naming convention, as long as they contain a .wav file.
Each .wav file is processed in 10-second chunks, transcribed, and saved as .txt next to the audio.

##💡 Notes
    1.Skips already transcribed languages if filtered in main().
    2.Auto-resamples non-16kHz audio.
    3.Very short chunks (< 4000 samples) are ignored to improve quality.
2. Alignment with align.py
This script aligns full audio to sentence-level timestamps using Facebook's MMS aligner.

##📂 Input Structure
Transcribed .txt and corresponding .wav files should be side by side:
```text
root_dir/
├── Hindi/
│ └── MKB_120_March_2025/
│     └── MKB_120_March_2025.wav
|     └── MKB_120_March_2025.txt
├── Assamese/
│ └──MKB_120_March_2025/
│   └── MKB_120_March_2025.wav
|     └── MKB_120_March_2025.txt
...
```
▶️ Run Alignment
```text
python align.py --root_dir /path/to/root_dir
```
##📎 License
```text
This project builds on open-source tools and respects their respective licenses:
Meta MMS — Licensed under CC-BY-NC 4.0
Hugging Face Transformers — Licensed under Apache 2.0
TorchAudio — Licensed under BSD-style license
```
⚖️ Please consult the individual project repositories for detailed licensing terms before distributing or using the derived outputs in commercial settings.
Let me know if you'd like me to:
- Add sample outputs or badges (like Python version, HuggingFace model version).
- Provide a `requirements.txt` template.
- Zip this with code into a downloadable project structure.

I'm happy to assist further.
