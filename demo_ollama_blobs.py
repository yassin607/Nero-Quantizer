import numpy as np
import os
from gguf import GGUFReader
from nero_core import NeroQuantizer # Assuming your core is in nero_core.py

def find_largest_blob(directory):
    """Locates the largest file in the Ollama blobs directory (usually the model)."""
    if not os.path.exists(directory):
        return None
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    if not files: return None
    return max(files, key=os.path.getsize)

def run_nero_demo():
    # Standard Ollama blobs path on Windows
    # Users can change this to their specific path
    blobs_path = os.path.expanduser(r"~\.ollama\models\blobs")

    try:
        model_path = find_largest_blob(blobs_path)
        if not model_path:
            print(f"❌ Ollama blobs not found at {blobs_path}. Please update the path in the script.")
            return

        print(f"🚀 Found Real Model: {os.path.basename(model_path)}")
        
        # 1. Open GGUF and extract the first heavy tensor (usually Embeddings)
        reader = GGUFReader(model_path)
        tensor = reader.tensors[0]
        weights = tensor.data.astype(np.float32)
        
        print(f"📦 Successfully Extracted: {tensor.name}")
        print(f"🔢 Weights Count: {weights.size:,}")

        # 2. Initialize NeroQuantizer (Higher precision for Embeddings)
        nq = NeroQuantizer(bits=4, block_size=32, clip_percentile=99.9)
        
        print("\n⚡ Quantizing Layer... Please wait...")
        quantized, scales, shape, padding = nq.fit_and_quantize(weights)
        
        # 3. Bit-Packing (VRAM-optimized storage)
        packed_data = nq.pack_weights(quantized)
        
        # 4. Verify Accuracy (MAE)
        recovered = nq.dequantize(quantized, scales, shape, padding)
        mae = np.mean(np.abs(weights - recovered))

        # 5. Calculate Metrics
        orig_size_mb = weights.nbytes / (1024**2)
        comp_size_mb = (packed_data.nbytes + scales.nbytes) / (1024**2)

        print(f"\n--- 📊 NeroQuantizer Benchmark Report ---")
        print(f"✅ Original Layer Size: {orig_size_mb:.2f} MB")
        print(f"✅ Compressed Nero Size: {comp_size_mb:.2f} MB")
        print(f"🚀 Compression Ratio:   {orig_size_mb / comp_size_mb:.2f}x")
        print(f"📉 Reconstruction Error (MAE): {mae:.6f}")

        if mae < 0.05:
            print("\n🌟 [SUCCESS] High intelligence preserved. Ready for Inference!")
        else:
            print("\n⚠️ [ADVISE] Consider adjusting block_size for better precision.")

    except Exception as e:
        print(f"❌ Error during demo: {e}")

if __name__ == "__main__":
    run_nero_demo()