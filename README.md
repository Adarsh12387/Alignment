# 🗣️ Multilingual Audio Transcription using Facebook MMS

This repository provides a Python script to perform **automatic transcription** of multilingual `.wav` audio files using the [Facebook MMS (Massively Multilingual Speech) model](https://huggingface.co/facebook/mms-1b-all). It supports chunked transcription, full-file transcription, and timestamped output.

---

## 📌 Features

- 🔤 Supports **36 Indian languages** (via MMS-compatible tags)
- 🎧 Transcribes **long audio files** by splitting them into manageable chunks
- 🕐 Optionally includes **timestamps** for each chunk
- ⚡ Efficient batch processing over structured directories
- 📄 Output saved as `.txt` or `.tsv` (with timestamps)

---

## 📁 Project Structure

```
.
├── transcribe.py                # Main transcription script
├── README.md                    # Documentation
└── requirements.txt             # Dependencies (optional)
```

---

## 🧰 Requirements

- Python 3.8+
- PyTorch (with CUDA support if available)
- `transformers` (HuggingFace)
- `torchaudio`
- `tqdm`
- `pandas`

### Install dependencies

```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt`, use:

```bash
pip install torch torchaudio transformers tqdm pandas
```

---

## 🛠️ Usage

### 1. Prepare Your Directory Structure

Your audio files must be organized by language and episode:

```
root_dir/
├── Hindi/
│   ├── MKB_120_March_2025/
│   │   └── MKB_120_March_2025.wav
│   └── ...
├── Assamese/
│   └── MKB_119_February_2025/
│       └── MKB_119_February_2025.wav
...
```

Each `.wav` file will be transcribed into a `.txt` file in the same folder.

### 2. Run the Script

```bash
python transcribe.py --root_dir /path/to/root_dir
```

The script will:

- Skip already-transcribed languages (you can modify the filter list)
- Automatically detect language code using `lang_map`
- Process audio in chunks (default: 10 seconds)
- Save output `.txt` files alongside the original `.wav`

---

## ⚙️ Optional Transcription Modes

### Transcribe Audio in One Shot (for short files)

```python
transcribe_audio_at_once_large(
    processor, model, language_code, logger,
    audio_path, output_txt_path, sampling_rate=16000
)
```

### Transcribe with Timestamps

```python
transcribe_audio_with_timestamps(
    processor, model, language_code, logger,
    audio_path, output_path, chunk_duration_sec=10
)
```

Output format (TSV):

| start_time | end_time | transcript |
|------------|----------|------------|
| 0.00       | 10.00    | Hello...   |

---

## 🌐 Supported Languages

Language codes used are adapted for compatibility with MMS tokenizers:

| Language  | Code                |
|-----------|---------------------|
| Hindi     | `hin`               |
| Urdu      | `urd-script_arabic` |
| Bengali   | `ben`               |
| Assamese  | `asm`               |
| Tamil     | `tam`               |
| Telugu    | `tel`               |
| Kannada   | `kan`               |
| Malayalam | `mal`               |
| Gujarati  | `guj`               |
| Marathi   | `mar`               |
| Odia      | `ory`               |
| Punjabi   | `pan`               |
| Sanskrit  | `hin`               |
| ...       | ...                 |

See the `lang_map` dictionary in `transcribe.py` for the full mapping.

---

## 🧪 Example Output

```
✅ Transcription saved to: /data/MKB_120_March_2025/MKB_120_March_2025.txt
```

Or with timestamps:

```
✅ Timestamped transcription saved to: /data/MKB_120_March_2025/MKB_120_March_2025_timestamps.tsv
```

---

## 🧩 Customization

- 🔍 **Filter Languages**: Modify the `if language in [...]` block in `main()` to include or exclude specific languages.
- 🧠 **Model Checkpoints**: The default model is:

  ```python
  model_id = "facebook/mms-1b-all"
  ```

  You can change it to any other MMS checkpoint supported by Hugging Face.
- ⏱️ **Chunk Duration**: You can change the default chunk duration (10 seconds) in:

  ```python
  transcribe_audio(..., chunk_duration_sec=10)
  ```

---

## 📝 Notes

- Audio chunks shorter than ~4000 samples (≈0.25 sec) are skipped.
- Use `transcribe_audio()` for long files to prevent memory overflow.
- The language adapter must match the expected MMS tokenizer format (e.g., `urd-script_arabic`).

---

## 📎 License

This project uses open-source tools and follows the licensing terms of:

- Facebook’s MMS model
- Hugging Face Transformers
- PyTorch ecosystem

---

## 🤝 Acknowledgements

- [Meta AI – MMS Project](https://github.com/facebookresearch/fairseq/tree/main/examples/mms)
- [Hugging Face Transformers](https://huggingface.co/facebook/mms-1b-all)
- [TorchAudio](https://pytorch.org/audio/)
