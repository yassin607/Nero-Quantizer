import numpy as np
import torch
from gguf import GGUFReader
from safetensors.torch import save_file
from tqdm import tqdm
import os
import argparse
from nero_core import NeroQuantizer # Import the core logic

def main():
    parser = argparse.ArgumentParser(description="Nero-V2.1: High-Efficiency GGUF to Nero-Safetensors Converter")
    parser.add_argument("--input", type=str, required=True, help="Path to the GGUF blob file")
    parser.add_argument("--output", type=str, default="Nero_Model_V2.1.safetensors", help="Output filename")
    parser.add_argument("--block_size", type=int, default=64, help="Block size for quantization (default: 64)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found at {args.input}")
        return

    print(f"🚀 Launching Nero-V2.1 (GGUF Compatibility Mode)...")
    
    reader = GGUFReader(args.input)
    nq = NeroQuantizer(bits=4, block_size=args.block_size, clip_percentile=99.9)
    nero_model_dict = {}

    # Process tensors directly from GGUF to bypass header errors
    for tensor in tqdm(reader.tensors, desc="Nero Processing"):
        name = tensor.name
        w_np = tensor.data.astype(np.float32)
        
        # Keep small tensors, norms, and biases in FP16 for accuracy
        if w_np.size < 2048 or "norm" in name or "bias" in name:
            nero_model_dict[name] = torch.from_numpy(w_np.astype(np.float16))
        else:
            # Apply Nero Quantization
            q_weights, scales, shape, pad_size = nq.fit_and_quantize(w_np)
            packed_weights = nq.pack_weights(q_weights)
            
            # Store compressed data
            nero_model_dict[f"{name}.packed"] = torch.from_numpy(packed_weights)
            nero_model_dict[f"{name}.scales"] = torch.from_numpy(scales)

    print(f"💾 Saving to {args.output}...")
    save_file(nero_model_dict, args.output)

    final_size_gb = os.path.getsize(args.output) / (1024**3)
    print(f"✅ MISSION ACCOMPLISHED!")
    print(f"📦 Nero-V2.1 Final Size: {final_size_gb:.2f} GB")

if __name__ == "__main__":
    main()