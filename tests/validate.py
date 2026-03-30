"""
TurboQuant JAX validation: compare attention scores on real model data.
Loads model via PyTorch (for HF compatibility), performs KV cache compression and
scoring evaluation via JAX to validate the JAX algorithms correctness out-of-core.
"""

import time
import os
import sys
import importlib.util
import numpy as np
import jax
import jax.numpy as jnp

try:
    from jax.experimental.compilation_cache import compilation_cache as cc

    # Setup JAX compilation cache for fast re-runs.
    cc.initialize_cache(os.path.expanduser("~/.jax_cache"))
except Exception:
    cc = None

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
HAS_BITSANDBYTES = importlib.util.find_spec("bitsandbytes") is not None

# Allow running as `python tests/validate.py` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant_jax.compressors import TurboQuantCompressorV2JAX, TurboQuantCompressorMSEJAX

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
NEEDLE = "The secret project code name is AURORA-7749."
QUESTION = "What is the secret project code name?"

FILLER = """The quarterly financial review meeting covered several topics including
budget allocations for the upcoming fiscal year, departmental spending reports, and projected
revenue streams from various business units. The committee discussed infrastructure upgrades
planned for the western regional offices and noted that maintenance schedules should be
coordinated with the facilities management team. Several action items were assigned to team
leads for follow-up before the next meeting cycle.\n\n"""


def build_prompt(tokenizer, target_tokens=2048, needle_pos=0.5):
    filler_len = len(tokenizer.encode(FILLER))
    n_reps = max(1, target_tokens // filler_len)
    needle_idx = int(n_reps * needle_pos)
    parts = []
    for i in range(n_reps):
        if i == needle_idx:
            parts.append(f"\n--- Memo ---\n{NEEDLE}\n--- End ---\n\n")
        parts.append(FILLER)
    haystack = "".join(parts)
    return f"<|im_start|>user\n{haystack}\nQuestion: {QUESTION}<|im_end|>\n<|im_start|>assistant\n"


def cpu_numpy(tensor):
    return tensor.detach().cpu().numpy()


def _block_tree(value):
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _extract_kv_layers(cache_obj):
    # transformers Cache object path (newer versions)
    if hasattr(cache_obj, "layers"):
        return [(layer.keys, layer.values) for layer in cache_obj.layers]

    # legacy tuple/list path: ((k, v), (k, v), ...)
    if isinstance(cache_obj, (list, tuple)):
        layers = []
        for layer in cache_obj:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                layers.append((layer.keys, layer.values))
                continue
            if isinstance(layer, (list, tuple)) and len(layer) >= 2:
                layers.append((layer[0], layer[1]))
                continue
            raise TypeError(f"Unsupported layer format: {type(layer)}")
        return layers

    raise TypeError(f"Unsupported cache type: {type(cache_obj)}")


def main():
    print("Loading model via PyTorch...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    load_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    if BitsAndBytesConfig is not None and HAS_BITSANDBYTES:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        print("bitsandbytes not available; loading in fp16 fallback mode.", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    model.eval()
    if torch.cuda.is_available():
        print(f"Loaded. Torch GPU Mem: {torch.cuda.memory_allocated() // 1024 // 1024} MB\n")
    
    # Note: JAX and PyTorch might compete for GPU memory.
    # By default JAX allocates 90% of GPU memory unless XLA settings are changed.
    print(f"JAX Backends: {jax.devices()}")

    for target_tokens in [2048, 4096, 8192]:
        prompt = build_prompt(tokenizer, target_tokens)
        
        # Determine device for PyTorch inputs
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=target_tokens + 256).to(torch_device)
        seq_len = inputs["input_ids"].shape[1]

        # Find needle token positions
        needle_phrase = "AURORA-7749"
        needle_tokens = tokenizer.encode(needle_phrase, add_special_tokens=False)
        input_ids_list = inputs["input_ids"][0].tolist()
        needle_start = None
        for i in range(len(input_ids_list) - len(needle_tokens) + 1):
            if input_ids_list[i:i + len(needle_tokens)] == needle_tokens:
                needle_start = i
                break
        if needle_start is None:
            for width in range(len(needle_tokens), 0, -1):
                sub = needle_tokens[:width]
                for i in range(len(input_ids_list) - width + 1):
                    if input_ids_list[i:i + width] == sub:
                        needle_start = i
                        break
                if needle_start is not None:
                    break

        print(f"{'=' * 70}")
        print(f"Context: {seq_len} tokens | Needle at token {needle_start}")
        print(f"{'=' * 70}")

        # Forward pass - capture KV cache
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True, output_attentions=False)
        cache = outputs.past_key_values
        kv_layers = _extract_kv_layers(cache)
        n_layers = len(kv_layers)
        
        for bits in [2, 3, 4]:
            total_compressed_bytes = 0
            total_uncompressed_bytes = 0
            top1_matches = 0
            top5_matches = 0
            needle_rank_sum = 0
            cosine_sims = []
            n_checks = 0

            # JAX loop
            for layer_idx, (keys_pt, values_pt) in enumerate(kv_layers):
                # Transfer PT to JAX
                keys = jnp.array(cpu_numpy(keys_pt), dtype=jnp.float16)
                values = jnp.array(cpu_numpy(values_pt), dtype=jnp.float16)

                B, H, S, D = keys.shape

                # Compress keys and values using JAX Compressor
                key_comp = TurboQuantCompressorV2JAX(D, bits, seed=layer_idx * 1000)
                val_comp = TurboQuantCompressorMSEJAX(D, bits, seed=layer_idx * 1000 + 500)

                compressed_k = key_comp.compress(keys)
                compressed_v = val_comp.compress(values)

                # Wait for execution since JAX is async
                _block_tree((compressed_k, compressed_v))

                k_bits = compressed_k["indices"].size * 8
                k_bits += compressed_k["qjl_signs"].size * 8
                k_bits += compressed_k["residual_norm"].size * 16
                k_bits += compressed_k["vec_norms"].size * 16

                v_bits = compressed_v["indices"].size * 8
                v_bits += compressed_v["vec_norms"].size * 16

                total_compressed_bytes += (k_bits + v_bits) / 8
                total_uncompressed_bytes += (keys.size + values.size) * 2  # fp16

                query = keys[:, :, -1:, :]  # (B, H, 1, D) - last token

                query_f32 = query.astype(jnp.float32)
                keys_f32 = keys.astype(jnp.float32)

                # Real scores
                real_scores = jnp.einsum("b h n d, b h s d -> b h n s", query_f32, keys_f32).squeeze(-2)  # (1, H, S)
                
                # TurboQuant scores
                tq_scores = key_comp.asymmetric_attention_scores(query, compressed_k).squeeze(-2)  # (1, H, S)

                # Wait evaluating similarity
                real_scores_np = np.array(real_scores)
                tq_scores_np = np.array(tq_scores)

                # Per-head comparison in numpy
                for h in range(H):
                    rs = real_scores_np[0, h]
                    ts = tq_scores_np[0, h]

                    # Cosine sim
                    dot = np.dot(rs, ts)
                    norm_rs = np.linalg.norm(rs)
                    norm_ts = np.linalg.norm(ts)
                    cos = dot / (norm_rs * norm_ts + 1e-12)
                    cosine_sims.append(cos)

                    # Top-1 match
                    real_top1 = np.argmax(rs)
                    tq_top1 = np.argmax(ts)
                    if real_top1 == tq_top1:
                        top1_matches += 1

                    # Top-5 match
                    tq_top5 = np.argsort(ts)[-5:]
                    if real_top1 in tq_top5:
                        top5_matches += 1

                    # Where does the needle rank?
                    if needle_start is not None:
                        rankings = np.argsort(ts)[::-1]
                        needle_rank = np.where(rankings == needle_start)[0]
                        if len(needle_rank) > 0:
                            needle_rank_sum += needle_rank[0].item()

                    n_checks += 1

            # Summary for this bit-width
            ratio = total_uncompressed_bytes / total_compressed_bytes
            avg_cos = sum(cosine_sims) / len(cosine_sims)
            top1_pct = 100 * top1_matches / n_checks
            top5_pct = 100 * top5_matches / n_checks
            avg_needle_rank = needle_rank_sum / n_checks if needle_start is not None else -1

            print(f"\n  TQ-{bits}bit:")
            print(f"    Compression:       {ratio:.1f}x  ({total_compressed_bytes / 1024 / 1024:.1f} MB vs {total_uncompressed_bytes / 1024 / 1024:.1f} MB)")
            print(f"    Score cosine sim:  {avg_cos:.6f}  (1.0 = perfect)")
            print(f"    Top-1 match:       {top1_pct:.1f}%  ({top1_matches}/{n_checks} heads)")
            print(f"    Top-5 match:       {top5_pct:.1f}%  ({top5_matches}/{n_checks} heads)")
            if needle_start is not None:
                print(f"    Avg needle rank:   {avg_needle_rank:.1f}  (lower = better, 0 = top)")

        print()

    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
