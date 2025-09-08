#!/usr/bin/env python3
"""
Checkpoint → RoMaTRTCoreFull remapper + diagnostics.

- Keeps DINO/ViT keys (no filtering).
- Strips common Lightning/module wrappers.
- Tries direct matches, then suffix-based matches onto encoder.dino.*
- If nothing loads, prints top-level CKPT prefixes & a key sample to help mapping.
"""

from __future__ import annotations
import re
from typing import Dict, Tuple, List, Optional
import torch
from torch import nn
from collections import Counter

RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^module\."), ""),
    # Handle MatchAnything LoFTR checkpoint structure
    (re.compile(r"^matcher\.model\.encoder\.cnn\."), "encoder.cnn."),
    (re.compile(r"^matcher\.model\.encoder\."), "encoder."),
    (re.compile(r"^matcher\.model\."), ""),
    (re.compile(r"^matcher\."), ""),
    (re.compile(r"^model\."), ""),
    # Handle DINOv2 structure (if present)
    (re.compile(r"^backbone\."), "encoder.dino."),
    (re.compile(r"^vit\."), "encoder.dino."),
    (re.compile(r"^dino\."), "encoder.dino."),
    (re.compile(r"^encoder\.vit\."), "encoder.dino."),
    (re.compile(r"^encoder\.dino\."), "encoder.dino."),  # direct match
    (re.compile(r"^encoder\."), "encoder."),  # identity
]


def _apply_rules(k: str) -> str:
    out = k
    for pat, rep in RULES:
        out = pat.sub(rep, out)
    return out


def _suffix_map(
    remapped: Dict[str, torch.Tensor], msd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    dino_mkeys = [mk for mk in msd.keys() if mk.startswith("encoder.dino.")]
    dino_suffixes = [(mk, mk[len("encoder.dino.") :]) for mk in dino_mkeys]
    for ck, v in remapped.items():
        if ck in msd:
            out[ck] = v
            continue
        for full_mk, suf in dino_suffixes:
            if ck.endswith(full_mk) or ck.endswith(suf):
                out[full_mk] = v
                break
    return out


def _print_diagnostics(
    state: Dict[str, torch.Tensor], msd: Dict[str, torch.Tensor], sample_n: int = 25
):
    def top_prefixes(keys, depth=2):
        pref = [".".join(k.split(".")[:depth]) for k in keys]
        return Counter(pref).most_common(20)

    ck = list(state.keys())
    mk = list(msd.keys())
    print("\n[WEIGHTS][DIAG] 0 keys loaded. Prefix histogram (ckpt, depth=2):")
    for p, c in top_prefixes(ck, depth=2):
        print(f"    {p:40s} x{c}")
    print("[WEIGHTS][DIAG] Model prefixes (depth=2):")
    for p, c in top_prefixes(mk, depth=2):
        print(f"    {p:40s} x{c}")
    print("[WEIGHTS][DIAG] Sample CKPT keys:")
    for k in ck[:sample_n]:
        print(f"    - {k}")


def remap_and_load(
    model: nn.Module, ckpt_path: str, save_sanitized: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    print(f"[WEIGHTS] Loading ckpt: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("state_dict", raw)

    remapped: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        remapped[_apply_rules(k)] = v

    msd = model.state_dict()
    cand = _suffix_map(remapped, msd)

    loadable: Dict[str, torch.Tensor] = {}
    mism: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for k, v in cand.items():
        if k not in msd:  # guard
            continue
        if tuple(v.shape) == tuple(msd[k].shape):
            loadable[k] = v
        else:
            mism.append((k, tuple(v.shape), tuple(msd[k].shape)))

    missing = [k for k in msd.keys() if k not in loadable]
    unexpected = [k for k in state.keys() if _apply_rules(k) not in cand]

    model.load_state_dict(loadable, strict=False)

    print("\n[WEIGHTS] Summary")
    print(f"  loaded:                {len(loadable)} / {len(msd)}")
    print(f"  unexpected (ignored):  {len(unexpected)}")
    print(f"  missing (unfilled):    {len(missing)}")
    if mism:
        print(f"  shape mismatches:      {len(mism)}")
        for k, s, t in mism[:24]:
            print(f"    - {k}: {s} -> expected {t}")

    if len(loadable) == 0:
        _print_diagnostics(state, msd, sample_n=30)

    if save_sanitized:
        torch.save({"state_dict": loadable}, save_sanitized)
        print(f"[WEIGHTS] Saved sanitized subset to {save_sanitized}")

    return loadable
