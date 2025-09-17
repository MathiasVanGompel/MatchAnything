# Conversion_Tensorrt/full/ckpt_tools_plus.py
import re
import torch
from collections import defaultdict

def _unwrap_state_dict(sd):
    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        return sd['state_dict']
    return sd

def _dtype_like_model(t, model):
    # put loaded tensor on CPU, but dtype consistent with model's first param
    mdtype = next(model.parameters()).dtype
    if t.dtype != mdtype:
        t = t.to(dtype=mdtype)
    return t

def summarize_prefixes(sd, max_prefix_len=3):
    # e.g. 'matcher.model.decoder.layers.0.attn.q.weight' -> 'matcher.model.decoder'
    # handy for debugging what the ckpt actually contains
    ctr = defaultdict(int)
    for k in sd.keys():
        parts = k.split('.')
        pref = '.'.join(parts[:max_prefix_len])
        ctr[pref] += 1
    return sorted(ctr.items(), key=lambda x: (-x[1], x[0]))

def try_load_extra_heads(model, ckpt_path, verbose=True):
    """
    Best-effort remapping of decoder/refiner weights from a MatchAnything/RoMa-style ckpt
    into your TRT-friendly modules named 'decoder.*' and 'refiner.*'.
    Non-destructive: only loads when shapes match. Returns (loaded, total_candidates).
    """
    if ckpt_path is None:
        if verbose:
            print("[loader+heads] no extra ckpt given; skipping.")
        return 0, 0

    try:
        raw = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"[loader+heads] failed to load extra ckpt: {e}")
        return 0, 0

    sd = _unwrap_state_dict(raw)
    if not isinstance(sd, dict):
        print("[loader+heads] extra ckpt has no dict state.")
        return 0, 0

    # Common prefixes (MatchAnything & RoMa):
    # - matcher.model.decoder.*, matcher.model.refiner.*
    # - model.decoder.*, model.refiner.*
    # - roma.decoder.*, roma.refiner.*
    # - decoder.*, refiner.* (already matching)
    prefix_rules = [
        (r'^matcher\.model\.decoder\.', 'decoder.'),
        (r'^matcher\.decoder\.',        'decoder.'),
        (r'^model\.decoder\.',          'decoder.'),
        (r'^roma\.decoder\.',           'decoder.'),
        (r'^decoder\.',                 'decoder.'),

        (r'^matcher\.model\.refiner\.', 'refiner.'),
        (r'^matcher\.refiner\.',        'refiner.'),
        (r'^model\.refiner\.',          'refiner.'),
        (r'^roma\.refiner\.',           'refiner.'),
        (r'^refiner\.',                 'refiner.'),
    ]

    m_sd = model.state_dict()
    remapped = {}
    total_cands = 0
    for k, v in sd.items():
        # only look at decoder/refiner-ish keys
        if not any(x in k for x in ('decoder', 'refiner')):
            continue
        total_cands += 1
        newk = None
        for pat, repl in prefix_rules:
            if re.search(pat, k):
                newk = re.sub(pat, repl, k)
                break
        if newk is None:
            continue
        if newk in m_sd and m_sd[newk].shape == v.shape:
            remapped[newk] = _dtype_like_model(v, model)

    # Load remapped subset (strict=False keeps everything else as-is)
    loaded = 0
    if remapped:
        missing_before = set(m_sd.keys()) - set(remapped.keys())
        load_res = model.load_state_dict({**m_sd, **remapped}, strict=False)
        # count how many of our remapped keys took effect
        loaded = len(remapped)
        if verbose:
            print(f"[loader+heads] remapped {loaded} tensors for decoder/refiner "
                  f"from '{ckpt_path}'.")
            if getattr(load_res, 'missing_keys', None):
                print(f"[loader+heads] still missing: {len(load_res.missing_keys)} params")
    else:
        if verbose:
            print("[loader+heads] found 0 remappable decoder/refiner tensors (name/shape mismatch).")

    return loaded, total_cands

def dump_ckpt_keys(ckpt_path, max_show=200):
    raw = torch.load(ckpt_path, map_location='cpu')
    sd = _unwrap_state_dict(raw)
    keys = list(sd.keys())
    print(f"[dump] total keys: {len(keys)}")
    for k in keys[:max_show]:
        print(k)
    print("------ prefix histogram (first 3 levels) ------")
    for pref, n in summarize_prefixes(sd):
        print(f"{n:5d}  {pref}")
