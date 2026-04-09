import torch
import numpy as np
from safetensors.torch import load_file
import os
import argparse

def verify_reconstruction(model_path, block_size=64):
    """
    Validates that the Nero-compressed weights can be correctly unpacked
    and reconstructed into readable float values.
    """
    if not os.path.exists(model_path):
        print(f"❌ File not found: {model_path}")
        return

    print(f"🔍 Loading Nero Model: {model_path}...")
    tensors = load_file(model_path)
    
    # Locate the first packed layer to test
    target_layer = None
    for key in tensors.keys():
        if ".packed" in key:
            target_layer = key.replace(".packed", "")
            break
            
    if not target_layer:
        print("❌ No packed layers found! Ensure the model was compressed with Nero-V2.1.")
        return

    print(f"🧪 Verifying integrity for layer: {target_layer}")

    packed = tensors[f"{target_layer}.packed"]
    scales = tensors[f"{target_layer}.scales"]

    # --- De-quantization Process ---
    # Unpack two 4-bit integers from each uint8 byte
    high = (packed >> 4).to(torch.int8)
    low = (packed & 0x0F).to(torch.int8)
    
    # Interleave and flatten to restore original sequence
    unpacked = torch.stack([high, low], dim=-1).flatten()
    
    # Shift back from unsigned range (4-bit min_int is -8)
    unpacked = (unpacked.to(torch.float32) - 8) 
    
    # Re-apply block scales
    try:
        reshaped_unpacked = unpacked.reshape(-1, block_size)
        reconstructed = reshaped_unpacked * scales.to(torch.float32)
        
        print(f"✅ Reconstruction Successful!")
        print(f"📊 Sample Weights (First 5): {reconstructed.flatten()[:5].tolist()}")
        print(f"📉 Max Absolute Value in layer: {reconstructed.abs().max().item():.4f}")
        
    except Exception as e:
        print(f"❌ Shape mismatch: Ensure block_size matches the compression setting (Current: {block_size}).")
        print(f"Error details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Nero-V2.1 Weight Integrity")
    parser.add_argument("--model", type=str, required=True, help="Path to the .safetensors model")
    parser.add_argument("--block_size", type=int, default=64, help="Block size used during compression")
    args = parser.parse_args()

    verify_reconstruction(args.model, args.block_size)