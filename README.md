# 🤖 AI Model Compressor v2.0

**AO Model Compre** – A lightweight, GUI-based tool for compressing large AI models like LLaMA and Stable Diffusion. This app provides smart compression, live progress tracking, and detailed benchmarking – all in a clean, dark-themed interface.

> 🔒 **Commercial use is NOT permitted. You must purchase the full project for any commercial use.**

---

## ✨ Features

- 🧠 Auto-detects LLaMA and Stable Diffusion models
- 📦 Supports ZIP archives containing models
- 💡 Smart compression based on model type (GGUF, ONNX INT8)
- 🚀 Real-time progress bars with animated feedback
- ⚙️ Advanced settings: quality modes and compression levels
- 📊 Optional benchmarking and report generation (TXT, JSON, HTML)
- 🖥️ Clean, dark GUI built with `tkinter`

---

## 🚀 Getting Started

### Requirements
- Python 3.7 or higher
- `tkinter` (usually included with Python)

### Installation
```bash
git clone https://github.com/gdsgbxfhdfhj/-ai_compressor.git
cd -ai_compressor
pip install -r requirements.txt
python model_compressor.py


---

🧩 Supported Models

🐑 LLaMA

Detects: config.json, pytorch_model.bin

Converts to GGUF format

Expected size reduction: 60–80%


🎨 Stable Diffusion

Detects: unet/, text_encoder/, vae/

Converts to ONNX with INT8 quantization

Expected size reduction: 50–70%



---

📁 Files in This Project

ai_model_compressor/
├── model_compressor.py     # Main Python GUI + logic
├── requirements.txt        # Required libraries
├── README.md               # Project overview (this file)


---

🔒 License

This project is under a Custom Non-Commercial License.

✅ Free for personal, educational, and research use

❌ Commercial use is strictly prohibited (e.g., in paid tools, APIs, SaaS, company products)


📩 To purchase the full project for commercial use, contact:
zamanikasra814@gmail.com

See full terms in the LICENSE file (available upon request).


---

🙏 Credits

🤗 Hugging Face

🔥 PyTorch

🧠 ONNX

💻 Open Source Developers


Created with ❤️ by Kasra Zamani — Promoting efficient, responsible AI.
