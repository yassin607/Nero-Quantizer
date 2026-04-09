import numpy as np
import torch

class NeroQuantizer:
    """
    NeroQuantizer: A high-performance block-wise quantization engine.
    Achieves ultra-low bitrates while maintaining model weights integrity.
    """
    def __init__(self, bits=4, block_size=64, clip_percentile=99.9):
        self.bits = bits
        self.block_size = block_size
        self.clip_p = clip_percentile
        self.max_int = (2**(bits-1)) - 1
        self.min_int = -(2**(bits-1))

    def fit_and_quantize(self, weights):
        """Quantizes weights into int8 blocks with scale factors."""
        flat_w = weights.flatten()
        pad_size = (self.block_size - len(flat_w) % self.block_size) % self.block_size
        padded_w = np.append(flat_w, np.zeros(pad_size, dtype=np.float32))
        blocks = padded_w.reshape(-1, self.block_size)
        
        # Calculate scales per block using percentile clipping
        scales = np.percentile(np.abs(blocks), self.clip_p, axis=1, keepdims=True)
        scales = np.maximum(scales, 1e-5) # Prevent division by zero
        
        final_scales = (scales / self.max_int).astype(np.float16)
        
        # Quantize process
        q_blocks = np.round(blocks / final_scales.astype(np.float32))
        q_blocks = np.nan_to_num(q_blocks)
        q_blocks = np.clip(q_blocks, self.min_int, self.max_int).astype(np.int8)
        
        return q_blocks, final_scales, weights.shape, pad_size

    def pack_weights(self, q_blocks):
        """Packs two 4-bit weights into a single uint8 byte for 50% extra compression."""
        unsigned_weights = (q_blocks - self.min_int).astype(np.uint8)
        packed = (unsigned_weights[:, 0::2] << 4) | (unsigned_weights[:, 1::2])
        return packed

    def dequantize(self, q_blocks, scales, original_shape, pad_size):
        """Reconstructs weights back to float32 (for inference)."""
        deq_blocks = q_blocks.astype(np.float32) * scales.astype(np.float32)
        flat_w = deq_blocks.flatten()
        
        if pad_size > 0:
            flat_w = flat_w[:-pad_size]
            
        return flat_w.reshape(original_shape)