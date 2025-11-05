# DeepSeek OCR PDF to Markdown Converter
This repository provides a tool to convert PDF documents into Markdown format using OCR (Optical Character Recognition) technology powered by DeepSeek OCR. It is designed to handle scanned PDFs and extract text content efficiently.

The output markdown files are structured with metadata. 
```yaml
page: 1
total_pages: 5
source: document.pdf
processor: DeepSeek-OCR
```

# Installation

```bash
uv venv
source .venv/bin/activate
# Until v0.11.1 release, you need to install vLLM from nightly build
uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly "triton-kernels@git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels"
pip install -r requirements.txt
```