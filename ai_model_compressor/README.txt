---
title: {{AI Model Compressor}}
emoji: {{🤖}}
colorFrom: {{gray}}
colorTo: {{blue}}
sdk: {{docker}}
sdk_version: "{{1.0}}"
app_file: model_compressor.py
pinned: false
---
title: AO model compre
# 🤖 AI Model Compressor v2.0 A professional-grade GUI tool for compressing AI models including LLaMA and Stable Diffusion models. This application provides intelligent compression algorithms, performance benchmarking, and detailed reporting capabilities. --- ## ✨ Features ### 🎯 Core Functionality - **Automatic Model Detection**: Intelligently identifies LLaMA and Stable Diffusion models - **Multi-Format Support**: Handles ZIP archives containing AI models - **Smart Compression**: Uses model-specific compression strategies - **Real-time Progress**: Live updates during compression process - **Batch Processing**: Process multiple models efficiently ### 🖥️ User Interface - **Modern Dark Theme**: Professional and eye-friendly interface - **Tabbed Design**: Organized workflow with Main, Settings, and Results tabs - **Interactive Controls**: Intuitive file selection and configuration - **Progress Visualization**: Animated progress bars with detailed status ### ⚙️ Advanced Settings - **Quality Modes**: - Fast (Lower Quality) - Balanced (Recommended) - High Quality (Slower) - **Compression Levels**: 1-8 scale (1=Fast, 8=Maximum compression) - **Benchmarking Options**: Optional performance testing - **Report Generation**: Detailed analysis and metrics ### 📊 Results & Analytics - **Compression Metrics**: Size reduction ratios and space savings - **Performance Benchmarks**: Inference speed and memory usage - **Export Options**: TXT, JSON, and HTML report formats - **System Analysis**: Hardware and software environment details --- ## 🚀 Quick Start ### Prerequisites - Python 3.7 or higher - tkinter (usually included with Python) ### Installation 1. **Clone the repository:** ```bash git clone https://github.com/yourusername/ai-model-compressor.git cd ai-model-compressor 

Install required dependencies:

pip install -r requirements.txt 

Run the application:

python model_compressor.py 

📦 Dependencies

Required

tkinter - GUI framework (usually included with Python)

zipfile - Archive handling

threading - Background processing

json - Data serialization

Optional (Enhanced Features)

torch - PyTorch model support

transformers - Hugging Face model support

huggingface_hub - Model downloading

psutil - System monitoring

🎮 Usage Guide

Step 1: Model Selection

Launch the application

Click "Browse" to select your ZIP file containing the AI model

The application will automatically detect the model type

Step 2: Configure Settings

Navigate to the "Advanced Settings" tab

Choose your quality mode: 

Fast: Quick compression with moderate quality loss

Balanced: Good balance of speed and quality (recommended)

High Quality: Best quality preservation but slower

Adjust compression level (1–8)

Enable benchmarking if desired

Step 3: Compression

Return to the "Main" tab

Click "🚀 Start Auto Compression"

Monitor progress in real-time

View results when complete

Step 4: Analysis

Check the "Results" tab for detailed metrics

Run benchmarks to test performance

Export reports for documentation

🔧 Supported Model Types

LLaMA Models

Detection: Identifies config.json, pytorch_model files

Compression: Downloads optimized GGUF versions

Expected Reduction: 60–80% size reduction

Format: Converts to GGUF format for efficiency

Stable Diffusion Models

Detection: Identifies unet/, text_encoder/, vae/ directories

Compression: ONNX conversion with INT8 quantization

Expected Reduction: 50–70% size reduction

Format: Optimized ONNX models

📊 Performance Metrics

Original vs compressed file sizes

Compression ratio (e.g., 3.2x)

Space saved percentage

Model loading time

Inference speed

Memory usage

Throughput measurements

🛠️ Technical Architecture

EnhancedModelCompressor: GUI and user interaction controller

ModelAnalyzer: Model structure analyzer

CompressionEngine: Handles quantization & conversion

BenchmarkSuite: Runs speed/memory benchmarks

ReportGenerator: Exports metrics in multiple formats

📁 Project Structure

ai-model-compressor/ ├── model_compressor.py ├── requirements.txt ├── README.md ├── LICENSE ├── docs/ │ └── ... ├── tests/ ├── examples/ 

🔒 Security & Privacy

All processing happens locally

No model data is transmitted

ZIP files are validated before use

Temporary files are cleaned automatically

🚨 Troubleshooting

See error messages in the terminal or console. For common fixes, refer to the docs/ folder.

🤝 Contributing

Pull requests are welcome! See CONTRIBUTING.md for details.

📄 License

This project is released under a Custom Non-Commercial License.

✅ Free for personal, research, and academic use.

❌ Commercial usage (e.g., in paid products, APIs, companies, or hosting platforms) requires written permission.

📩 Contact for licensing zamanikasra814@gmail.con

Full terms available in the LICENSE file.

🙏 Acknowledgments

Hugging Face

PyTorch

ONNX Community

Open Source Contributors

Made with ❤️ by Kasra Zamani – For responsible and ethical AI development 