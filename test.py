import os
import numpy as np
from gguf import GGUFReader

# 1. تحديد فولدر الـ blobs الأساسي بتاع Ollama
blobs_path = r"C:\Users\hp\.ollama\models\blobs"

def find_largest_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    if not files:
        return None
    # بيرجع أكبر ملف في الفولدر (غالباً بيكون الموديل)
    return max(files, key=os.path.getsize)

try:
    model_path = find_largest_file(blobs_path)
    
    if model_path:
        print(f"--- Found Model File: {os.path.basename(model_path)} ---")
        print(f"--- Size: {os.path.getsize(model_path) / (1024**3):.2f} GB ---")
        
        # 2. محاولة فتح الملف
        reader = GGUFReader(model_path)
        
        # نجيب أول Tensor (طبقة) نجرب عليها
        tensor = reader.tensors[0]
        print(f"\nSuccessfully opened layer: {tensor.name}")
        print(f"Shape: {tensor.shape}")
        
        # تحويل البيانات لـ NumPy عشان NeroQuantizer
        weights = tensor.data.astype(np.float32)
        print(f"Weights extracted! Ready for NeroQuantizer.")
        
    else:
        print("No files found in the blobs directory.")

except Exception as e:
    print(f"Error: {e}")