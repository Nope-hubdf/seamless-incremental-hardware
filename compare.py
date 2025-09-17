#!/usr/bin/env python3

import argparse, os, gc, json, random, csv
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from matplotlib import pyplot as plt

# Optional, used only when --engine vllm
try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

SUFFIX = "Let's think step by step and answer in \\boxed{}."

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _get_base_module_and_head(model: AutoModelForCausalLM):
    head = getattr(model, "lm_head", None) or getattr(model, "embed_out", None)
    base = None
    base_prefix = getattr(model, "base_model_prefix", None)
    if isinstance(base_prefix, str) and hasattr(model, base_prefix):
        base = getattr(model, base_prefix)
    for name in ["model", "transformer", "language_model", "backbone", "base_model"]:
        if base is None and hasattr(model, name):
            base = getattr(model, name)
    if base is None or head is None:
        raise RuntimeError("Could not locate base transformer or lm_head on the HF model.")
    return base, head

@torch.no_grad()
def _chunked_token_logprobs_from_hidden(hiddens: torch.Tensor,
                                        head_weight: torch.Tensor,
                                        targets: torch.Tensor,
                                        time_chunk: int = 256) -> torch.Tensor:
    """
    hiddens: [B, L, H] (on CPU)
    head_weight: [V, H] on its own (likely CUDA) device
    targets: [B, L-1] (on CPU)
    returns lp: [B, L-1] (float32, on CPU)
    """
    B, L, H = hiddens.shape
    T = L - 1
    V, Hw = head_weight.shape
    assert H == Hw, f"Hidden size mismatch: {H} vs {Hw}"

    out_device = hiddens.device              # CPU
    weight_device = head_weight.device       # e.g., cuda:0/1

    # Ensure we run on the correct CUDA device to avoid cross-device kernel weirdness
    if weight_device.type == 'cuda':
        torch.cuda.set_device(weight_device.index)

    # keep W on the head device, dtype matched with hiddens (bf16 on CPU is fine)
    W = head_weight.to(dtype=hiddens.dtype, device=weight_device)

    lp = torch.empty((B, T), dtype=torch.float32, device=out_device)

    t = 0
    cur_chunk = max(1, int(time_chunk))

    while t < T:
        cur = min(cur_chunk, T - t)
        try:
            # Slice, make contiguous, then copy CPU->CUDA (blocking copy for stability)
            h = hiddens[:, t:t+cur, :].contiguous().view(-1, H).to(weight_device, non_blocking=False)
            y = targets[:, t:t+cur].contiguous().view(-1).to(weight_device, non_blocking=False)

            logits = F.linear(h, W)  # [B*cur, V] on weight_device

            # Numerically stable log softmax for chosen tokens
            m = logits.max(dim=-1).values
            lse = torch.logsumexp((logits - m.unsqueeze(1)).to(torch.float32), dim=-1) + m.to(torch.float32)
            chosen = logits.gather(1, y.unsqueeze(1)).squeeze(1).to(torch.float32)

            lp_chunk = (chosen - lse).view(B, cur).to(out_device, non_blocking=False)
            lp[:, t:t+cur] = lp_chunk

            # cleanup
            del h, y, logits, m, lse, chosen, lp_chunk
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            t += cur
            cur_chunk = time_chunk
        except RuntimeError as e:
            # Back off time chunk on OOM
            if "out of memory" in str(e).lower() and cur > 1:
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                cur_chunk = max(1, cur // 2)
                continue
            raise
    return lp

def infer_log_probs_batch(model: AutoModelForCausalLM, sequences, device_hint: str, time_chunk: int = 256):
    """
    Run base forward (sharded on GPUs), offload [B,L,H] to CPU, then do chunked head on the head device.
    """
    use_cpu_io = hasattr(model, "hf_device_map") or hasattr(model, "device_map")
    tgt_device = 'cpu' if use_cpu_io else device_hint

    lens = [len(s) for s in sequences]
    Lm = max(lens) if lens else 0
    pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else (model.config.eos_token_id or 0)

    inp = torch.full((len(sequences), Lm), pad_id, dtype=torch.long, device=tgt_device)
    for i, s in enumerate(sequences):
        if len(s) > 0:
            inp[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=tgt_device)
    attn = (inp != pad_id).long()

    try:
        model.config.use_cache = False
    except Exception:
        pass

    base, head = _get_base_module_and_head(model)

    # If the model was loaded with a device_map (even single GPU), ensure inputs
    # are moved to the same device as the base module's first parameter (embedding device)
    if use_cpu_io:
        try:
            first_param_device = next(base.parameters()).device
            if inp.device != first_param_device:
                inp = inp.to(first_param_device)
                attn = attn.to(first_param_device)
        except Exception:
            pass

    with torch.inference_mode():
        outputs = base(
            input_ids=inp,
            attention_mask=attn,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        hidden_states = outputs[0]  # [B, Lm, H] on last shard GPU
        del outputs

    # Offload activations to CPU, then make contiguous + (optional) pin
    if hidden_states.is_cuda:
        hidden_states = hidden_states.to('cpu', non_blocking=False)
        torch.cuda.synchronize(); torch.cuda.empty_cache()
    hidden_states = hidden_states.contiguous()
    try:
        hidden_states = hidden_states.pin_memory()
    except Exception:
        pass

    tgt = inp[:, 1:]  # [B, Lm-1] on CPU

    lp = _chunked_token_logprobs_from_hidden(
        hiddens=hidden_states,
        head_weight=head.weight,
        targets=tgt,
        time_chunk=time_chunk,
    )

    del hidden_states
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return lp, lens

# ------------------------- plotting -------------------------

def plot_correlation(eng_logp, hf_logp, out_png, log_space=False):
    p_min, p_max = (-40, 0) if log_space else (0, 1)
    X = (eng_logp if log_space else eng_logp.exp()).float().cpu().numpy()
    Y = (hf_logp  if log_space else hf_logp.exp()).float().cpu().numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [4, 2]})
    axes[0].set_aspect('equal')
    axes[0].set_xlim(p_min, p_max); axes[0].set_ylim(p_min, p_max)
    axes[1].set_xlim(p_min, p_max)
    hist, xe, ye = np.histogram2d(X, Y, bins=100, range=[[p_min, p_max], [p_min, p_max]], density=False)
    hist = np.log(hist + 1e-10)
    Xm, Ym = np.meshgrid(xe[:-1], ye[:-1])
    im = axes[0].pcolormesh(Xm, Ym, hist.T, shading='auto')
    axes[0].plot([p_min, p_max], [p_min, p_max], linestyle='--', linewidth=1)
    axes[0].set_xlabel('Engine ' + ('log-prob' if log_space else 'probability'))
    axes[0].set_ylabel('HF '     + ('log-prob' if log_space else 'probability'))
    fig.colorbar(im, ax=axes[0], label='Log Frequency')
    hx, xe1 = np.histogram(X, bins=100, range=[p_min, p_max], density=True)
    hy, ye1 = np.histogram(Y, bins=100, range=[p_min, p_max], density=True)
    axes[1].plot(xe1[:-1], np.log(hx + 1e-12), label='Engine')
    axes[1].plot(ye1[:-1], np.log(hy + 1e-12), label='HF')
    axes[1].legend(); axes[1].set_ylabel('Log Density'); axes[1].set_xlabel('log-prob' if log_space else 'probability')
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def plot_sample_prob_diff(hf_logp, eng_logp, out_png):
    diff = hf_logp.exp() - eng_logp.exp()
    xs = np.arange(len(diff))
    plt.figure(figsize=(10, 4))
    plt.plot(xs, diff.cpu().numpy())
    plt.xlabel('Response token index'); plt.ylabel('Δ prob (HF − Engine)')
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

# ------------------------- engines -------------------------

def run_vllm(prompt_ids_list, model, batch_size, max_new_tokens, seed=0, use_inductor=False):
    assert LLM is not None, 'vLLM is not installed.'
    llm = LLM(
        model=model, 
        dtype=torch.bfloat16, 
        trust_remote_code=True,
        compilation_config={
            "use_inductor": use_inductor,
        }
    )
    # NOTE: script not valid for temp != 1.0, vllm needs a patch bc logprobs are returned before sampling
    sp = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=int(max_new_tokens),
        logprobs=0,
        detokenize=True,
        seed=seed
    )
    prompt_ids, gen_ids, gen_logp, texts = [], [], [], []
    for i in range(0, len(prompt_ids_list), batch_size):
        batch_prompts = prompt_ids_list[i:i+batch_size]
        # vLLM expects a list of PromptInputs; use tokenized prompts schema
        batch_inputs = [{"prompt_token_ids": ids} for ids in batch_prompts]
        outs = llm.generate(batch_inputs, sampling_params=sp)
        for pid_sent, o in zip(batch_prompts, outs):
            sample = o.outputs[0]
            p_ids  = list(pid_sent)
            g_ids  = list(sample.token_ids)

            if sample.logprobs is None:
                raise RuntimeError("vLLM returned no logprobs; set SamplingParams.logprobs >= 1.")

            chosen_lp = []
            for t, tok_id in enumerate(g_ids):
                lp_dict = sample.logprobs[t]   # dict[token_id -> Logprob]
                lp_obj  = lp_dict.get(tok_id)
                if lp_obj is None:
                    raise RuntimeError("Chosen token not in returned top-k logprobs (???)")
                chosen_lp.append(float(lp_obj.logprob))
            g_lp = torch.tensor(chosen_lp, dtype=torch.float32)

            prompt_ids.append(p_ids); gen_ids.append(g_ids); gen_logp.append(g_lp); texts.append(sample.text)
    del llm; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return prompt_ids, gen_ids, gen_logp, texts

def run_sglang(prompt_ids_list, model, batch_size, max_new_tokens):
    from sglang.srt.entrypoints.engine import Engine
    engine = Engine(model_path=model, dtype='bfloat16', tp_size=1, trust_remote_code=True, load_format='auto', log_level='INFO', max_running_requests=1024)
    prompt_ids, gen_ids, gen_logp, texts = [], [], [], []
    for i in range(0, len(prompt_ids_list), batch_size):
        batch_ids = prompt_ids_list[i:i+batch_size]
        sp = {
            "n": 1, "max_new_tokens": int(max_new_tokens), "temperature": 1.0, "top_p": 1.0,
            "top_k": -1, "ignore_eos": False, "min_new_tokens": 0,
            "skip_special_tokens": True, "spaces_between_special_tokens": True,
        }
        outs = engine.generate(prompt=None, sampling_params=sp, return_logprob=True, input_ids=batch_ids, image_data=None)
        for pid, out in zip(batch_ids, outs):
            tpl = out["meta_info"]["output_token_logprobs"]
            g_lp = torch.tensor([float(t[0]) for t in tpl], dtype=torch.float32)
            g_ii = [int(t[1]) for t in tpl]
            prompt_ids.append(pid); gen_ids.append(g_ii); gen_logp.append(g_lp); texts.append(out["text"])
    try:
        engine.shutdown()
    except Exception:
        pass
    del engine; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return prompt_ids, gen_ids, gen_logp, texts

def build_hf_model(model_name):
    # Give accelerate a memory budget so it offloads instead of spiking one GPU
    max_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            giB = int((total * 0.95) // (1024**3))
            max_memory[i] = f"{giB}GiB"
    max_memory["cpu"] = "256GiB"

    # Flash-Attention 2 only
    hf = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        device_map='auto' if torch.cuda.is_available() else None,
        max_memory=max_memory if torch.cuda.is_available() else None,
    )
    hf.eval()
    try:
        hf.config.use_cache = False
    except Exception:
        pass
    if not torch.cuda.is_available():
        hf.to("cpu")
    return hf, ('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', choices=['vllm','sglang'], required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--hf-batch-size', type=int, default=64, help='Batch size for HF scoring (keep small to avoid OOM)')
    ap.add_argument('--time-chunk-size', type=int, default=128, help='Time chunk (tokens) for head projection')
    ap.add_argument('--max-new-tokens', type=int, default=32768)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=16)
    ap.add_argument('--out', default='out')
    ap.add_argument('--dataset', default='AI-MO/aimo-validation-aime') # or MathArena/aime_2025
    ap.add_argument('--vllm-use-inductor', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed); ensure_dir(args.out)

    if args.vllm_use_inductor:
        assert args.engine == 'vllm', 'vLLM with inductor requires vLLM engine'

    ds = load_dataset(args.dataset, split='train')
    user_texts = [f"{row['problem']}\n\n{SUFFIX}" for row in ds]
    user_texts = user_texts * args.n

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Build chat prompts using the model's chat template
    messages_list = [[{"role": "user", "content": text}] for text in user_texts]
    chat_prompt_ids = [
        tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_tensors=None)
        for msgs in messages_list
    ]
    # Ensure lists of ints for sglang
    chat_prompt_ids = [
        ids.tolist() if hasattr(ids, 'tolist') else (list(ids[0]) if isinstance(ids, (tuple, list)) and hasattr(ids[0], '__iter__') else list(ids))
        for ids in chat_prompt_ids
    ]

    if args.engine == 'vllm':
        p_ids, g_ids, g_lp, texts = run_vllm(chat_prompt_ids, args.model, args.batch_size, args.max_new_tokens, seed=args.seed, use_inductor=args.vllm_use_inductor)
    else:
        p_ids, g_ids, g_lp, texts = run_sglang(chat_prompt_ids, args.model, args.batch_size, args.max_new_tokens)

    hf, device = build_hf_model(args.model)

    # Sanity checks
    V = hf.get_input_embeddings().weight.size(0)
    for idx, (pi, gi, elp) in enumerate(zip(p_ids, g_ids, g_lp)):
        assert len(gi) == len(elp), f"Engine IDs/logprobs length mismatch at sample {idx} (len(ids)={len(gi)} vs len(lp)={len(elp)})"
        if (pi and max(pi) >= V) or (gi and max(gi) >= V):
            raise ValueError(f"Token id out of range at sample {idx}: vocab={V}, max_id={max(pi+gi)}")

    sequences = [pi + gi for pi, gi in zip(p_ids, g_ids)]

    max_pos = getattr(hf.config, "max_position_embeddings", None)
    if max_pos is not None:
        too_long = [i for i, s in enumerate(sequences) if len(s) > max_pos]
        if len(too_long) > 0:
            print(f"Warning: {len(too_long)} sequences exceed model max_position_embeddings={max_pos} and may be truncated or slow.")

    # ---------- HF scoring (length-bucket to cut padding) ----------
    idxs = list(range(len(sequences)))
    idxs.sort(key=lambda i: len(sequences[i]))  # shortest -> longest
    hf_rows = [None] * len(sequences)

    for start in range(0, len(sequences), args.hf_batch_size):
        batch_idx = idxs[start:start+args.hf_batch_size]
        seq_batch = [sequences[i] for i in batch_idx]

        lp_batch, lens_batch = infer_log_probs_batch(
            hf, seq_batch, device, time_chunk=args.time_chunk_size
        )
        for j, (i_orig, L) in enumerate(zip(batch_idx, lens_batch)):
            hf_rows[i_orig] = lp_batch[j, :L-1].detach().cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Align engine vs HF on generated tokens
    eng_slices, hf_slices, slice_indices = [], [], []
    for idx, (pi, gi, eng_lp) in enumerate(zip(p_ids, g_ids, g_lp)):
        Lp, Lg = len(pi), len(gi)
        hf_row = hf_rows[idx]
        start  = max(Lp - 1, 0)
        hf_slice = hf_row[start:start+Lg]
        m = min(len(eng_lp), len(hf_slice))
        if m > 0:
            eng_slices.append(eng_lp[:m])
            hf_slices.append(hf_slice[:m])
            slice_indices.append(idx)

    eng_all = torch.cat(eng_slices) if eng_slices else torch.empty(0)
    hf_all  = torch.cat(hf_slices)  if hf_slices  else torch.empty(0)

    # Save raw per-item
    # Re-ensure output directory and use absolute path to avoid cwd ambiguity
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, 'engine_outputs.jsonl')
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for pi, gi, elp, txt in zip(p_ids, g_ids, g_lp, texts):
            f.write(json.dumps({'prompt_ids': pi, 'gen_ids': gi, 'gen_logprobs': [float(x) for x in elp.tolist()], 'text': txt})+'\n')

    # Summary metrics + plots
    if len(eng_all) > 0:
        e = eng_all.float().cpu().numpy()
        h = hf_all.float().cpu().numpy()
        mae = float(np.mean(np.abs(h - e)))
        rmse = float(np.sqrt(np.mean((h - e) ** 2)))
        corr = float(np.corrcoef(h, e)[0, 1]) if len(h) > 1 else float('nan')
        # Additional metrics in probability space
        lnp = np.clip(h.astype(np.float64), -80.0, 0.0)
        lnq = np.clip(e.astype(np.float64), -80.0, 0.0)
        p_raw = np.exp(lnp)
        q_raw = np.exp(lnq)
        diff = p_raw - q_raw
        rollout_probs_diff_mean = float(np.mean(diff))
        rollout_probs_diff_std = float(np.std(diff))
        # Bernoulli KL between chosen-token probabilities (clip to avoid log(0))
        eps = 1e-12
        p = np.clip(p_raw, eps, 1.0 - eps)
        q = np.clip(q_raw, eps, 1.0 - eps)
        kl_vals = p * (np.log(p) - np.log(q)) + (1.0 - p) * (np.log1p(-p) - np.log1p(-q))
        kl_divergence = float(np.mean(kl_vals))
        # Completion length stats (in tokens)
        completion_lengths = np.array([len(gen_ids) for gen_ids in g_ids], dtype=np.int64)
        avg_completion_length = float(np.mean(completion_lengths)) if completion_lengths.size > 0 else 0.0
        min_completion_length = int(np.min(completion_lengths)) if completion_lengths.size > 0 else 0
        max_completion_length = int(np.max(completion_lengths)) if completion_lengths.size > 0 else 0
        with open(os.path.join(out_dir, 'summary_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({
                "mae_logprob": mae,
                "rmse_logprob": rmse,
                "pearson_r": corr,
                "kl_divergence": kl_divergence,
                "rollout_probs_diff_mean": rollout_probs_diff_mean,
                "rollout_probs_diff_std": rollout_probs_diff_std,
                "avg_completion_length": avg_completion_length,
                "min_completion_length": min_completion_length,
                "max_completion_length": max_completion_length,
                "n_tokens": int(len(h))
            }, f, indent=2)

        plot_correlation(eng_all, hf_all, os.path.join(out_dir, 'diff_raw.png'),  log_space=False)
        plot_correlation(eng_all, hf_all, os.path.join(out_dir, 'diff_log.png'),  log_space=True)

        j = len(eng_slices) // 2 if eng_slices else 0
        if eng_slices:
            plot_sample_prob_diff(hf_slices[j], eng_slices[j], os.path.join(out_dir, 'sample_prob_diff.png'))
            orig_j = slice_indices[j]
            with open(os.path.join(out_dir, 'sample_completion.txt'), 'w', encoding='utf-8') as f:
                f.write(texts[orig_j])
            toks = tokenizer.convert_ids_to_tokens(g_ids[orig_j][:len(eng_slices[j])])
            with open(os.path.join(out_dir, 'sample_token_diffs.csv'), 'w', newline='', encoding='utf-8') as cf:
                w = csv.writer(cf); w.writerow(['idx','token','prob_hf','prob_engine','delta'])
                for i, (hlp, elp, t) in enumerate(zip(hf_slices[j], eng_slices[j], toks)):
                    ph, pe = float(hlp.exp()), float(elp.exp())
                    w.writerow([i, t, ph, pe, ph-pe])

            j_longest = max(range(len(slice_indices)), key=lambda k: len(g_ids[slice_indices[k]]))
            orig_longest = slice_indices[j_longest]
            plot_sample_prob_diff(hf_slices[j_longest], eng_slices[j_longest], os.path.join(out_dir, 'longest_prob_diff.png'))
            with open(os.path.join(out_dir, 'longest_completion.txt'), 'w', encoding='utf-8') as f:
                f.write(texts[orig_longest])
            toks = tokenizer.convert_ids_to_tokens(g_ids[orig_longest][:len(eng_slices[j_longest])])
            with open(os.path.join(out_dir, 'longest_token_diffs.csv'), 'w', newline='', encoding='utf-8') as cf:
                w = csv.writer(cf); w.writerow(['idx','token','prob_hf','prob_engine','delta'])
                for i, (hlp, elp, t) in enumerate(zip(hf_slices[j_longest], eng_slices[j_longest], toks)):
                    ph, pe = float(hlp.exp()), float(elp.exp())
                    w.writerow([i, t, ph, pe, ph-pe])

    # Clean up torch.distributed process group if initialized (avoids NCCL warning)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass

    print('Saved to', out_dir)

if __name__ == '__main__':
    main()
