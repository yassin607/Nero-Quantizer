# 🌌 Nero-V2.1: Ultra-Low VRAM Quantization Engine

**Nero-V2.1** is a high-performance model compression toolkit designed for AI researchers and engineers who want to run massive LLMs on consumer hardware. 

Using custom **4-bit block-wise quantization** and **bit-shifting packing**, I successfully compressed a **Qwen2.5-14B model down to 2.43 GB**, achieving an incredible **11x compression ratio** with minimal intelligence loss.



## 🔥 Key Achievements
* **Compression Power:** 14B Parameters -> 2.43 GB Safetensors.
* **Zero-Header GGUF Support:** Directly extracts weights from Ollama/GGUF blobs.
* **Smart Clipping:** Uses 99.9th percentile clipping to preserve critical weights.
* **VRAM Optimized:** Packed 4-bit storage for ultra-low memory footprint.

## 🛠️ Installation
```bash
git clone [https://github.com/yassin607/Nero-Quantizer.git](https://github.com/yassin607/Nero-Quantizer.git)
cd Nero-Quantizer
pip install -r requirements.txt