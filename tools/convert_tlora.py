"""Convert a legacy T-LoRA checkpoint into a standard LoRA checkpoint.

Legacy T-LoRA checkpoints produced by an older version of ``lycoris.modules.tlora``
use the keys::

    <lora_name>.q_layer.weight
    <lora_name>.p_layer.weight
    <lora_name>.lambda_layer
    <lora_name>.alpha

These keys are not recognized by ComfyUI, which expects the standard LoRA
format ``lora_up.weight`` / ``lora_down.weight`` / ``alpha``.

T-LoRA's effective weight delta at inference time (with all ranks active,
i.e. no timestep masking) is::

    ΔW = P · diag(λ) · Q - P_base · diag(λ_base) · Q_base

The base matrices ``P_base``, ``diag(λ_base)``, ``Q_base`` come from an SVD
of the **original base model weight** of the wrapped layer — they were not
persisted to the legacy checkpoint.  This script reconstructs them by
running the same SVD against the base model state dict and then stacks the
trained and base paths into a single rank-``2r`` standard LoRA::

    lora_up   = [P · diag(λ), -P_base · diag(λ_base)]   shape (out, 2r)
    lora_down = [Q ; Q_base]                            shape (2r, in)
    alpha     = 2 * original_alpha

The factor-of-two on alpha keeps ``alpha / rank`` identical to the original
``alpha / lora_dim`` scaling T-LoRA uses.

Usage::

    python tools/convert_tlora.py \\
        --base-model /path/to/anima_base.safetensors \\
        --tlora /path/to/trained_tlora.safetensors \\
        --output /path/to/converted_lora.safetensors
"""

import argparse
import os
import sys
from collections import defaultdict
from typing import Dict, Optional

import torch


LORA_PREFIXES = ("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_")


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _save_state_dict(sd: Dict[str, torch.Tensor], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".safetensors"):
        from safetensors.torch import save_file

        save_file(sd, path)
    else:
        torch.save(sd, path)


def _build_base_lookup(
    base_sd: Dict[str, torch.Tensor],
    wanted_lora_names: "set[str]",
) -> Dict[str, str]:
    """Build a map from LyCORIS lora-name to the corresponding base weight key.

    LyCORIS names a module by joining its dotted module path with ``_`` and
    prefixing with e.g. ``lora_unet_``.  The base ``.safetensors`` file we are
    handed here is usually a full checkpoint whose weight keys carry an
    extra prefix such as ``model.diffusion_model.`` that the in-memory LyCORIS
    module never saw — the training loop wraps a sub-module, so only its
    ``named_modules()`` path ends up in the lora_name.

    We therefore match by **suffix**: for every candidate weight key in the
    base state dict, we take its dotted module path and progressively strip
    leading path segments, forming candidate lora names of the form
    ``<prefix>_<suffix_with_dots_as_underscores>``.  The first candidate
    that appears in ``wanted_lora_names`` wins.  We prefer the longest
    suffix (least prefix stripping) so that e.g. a module legitimately
    named ``model_diffusion_model_blocks_0_...`` would still be preferred
    over the truncated ``blocks_0_...`` alternative if it existed.
    """
    # Every prefix LyCORIS can emit.  We loop over all of them because the
    # user's checkpoint may contain multiple scopes.
    known_prefixes = ("lora_unet", "lora_te", "lora_te1", "lora_te2")

    lookup: Dict[str, str] = {}
    for key in base_sd.keys():
        if not key.endswith(".weight"):
            continue
        module_path = key[: -len(".weight")]
        parts = module_path.split(".")

        # Try progressively more aggressive prefix stripping: start with the
        # full path, then drop one leading segment at a time.
        found = False
        for start in range(len(parts)):
            suffix = "_".join(parts[start:])
            for prefix in known_prefixes:
                candidate = f"{prefix}_{suffix}"
                if candidate in wanted_lora_names:
                    # Prefer the first (longest) match, but don't overwrite an
                    # existing entry that was set from an even longer suffix
                    # for the same lora_name — that earlier entry was at least
                    # as specific.
                    if candidate not in lookup:
                        lookup[candidate] = key
                    found = True
                    break
            if found:
                break
    return lookup


def _group_tlora_keys(
    tlora_sd: Dict[str, torch.Tensor]
) -> Dict[str, Dict[str, torch.Tensor]]:
    groups: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
    legacy_suffixes = {
        "q_layer.weight",
        "p_layer.weight",
        "lambda_layer",
        "alpha",
    }
    for full_key, tensor in tlora_sd.items():
        for suffix in legacy_suffixes:
            marker = "." + suffix
            if full_key.endswith(marker):
                lora_name = full_key[: -len(marker)]
                groups[lora_name][suffix] = tensor
                break
    return groups


def _svd_base(
    base_weight: torch.Tensor,
    lora_dim: int,
    sig_type: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-derive (q_base, p_base, lambda_base) exactly as TLoraModule does.

    For conv layers TLoraModule SVDs the flattened weight of shape
    ``(out, in * k * k)`` but only keeps the first ``in`` columns of the
    right-singular vectors (the rest would describe the spatial kernel
    structure which TLoRA's 1x1 decomposition discards).  We must replicate
    that truncation here so the reconstructed base matches what TLoRA held
    in its ``base_q`` / ``base_p`` / ``base_lambda`` buffers at train time.
    """
    weight = base_weight.detach().to(device=device, dtype=torch.float32)
    is_conv = weight.dim() > 2
    if is_conv:
        out_dim = weight.shape[0]
        in_dim = weight.shape[1]
        weight_2d = weight.reshape(out_dim, -1)
    else:
        in_dim = weight.shape[1]
        weight_2d = weight

    u, s, vh = torch.linalg.svd(weight_2d, full_matrices=False)

    if sig_type == "principal":
        q_base = vh[:lora_dim]
        p_base = u[:, :lora_dim]
        lam_base = s[:lora_dim]
    elif sig_type == "last":
        q_base = vh[-lora_dim:]
        p_base = u[:, -lora_dim:]
        lam_base = s[-lora_dim:]
    elif sig_type == "middle":
        start_q = (vh.shape[0] - lora_dim) // 2
        start_p = (u.shape[1] - lora_dim) // 2
        start_s = (s.shape[0] - lora_dim) // 2
        q_base = vh[start_q : start_q + lora_dim]
        p_base = u[:, start_p : start_p + lora_dim]
        lam_base = s[start_s : start_s + lora_dim]
    else:
        raise ValueError(f"Unknown sig_type: {sig_type}")

    if q_base.shape[0] < lora_dim:
        pad_size = lora_dim - q_base.shape[0]
        q_base = torch.nn.functional.pad(q_base, (0, 0, 0, pad_size))
        p_base = torch.nn.functional.pad(p_base, (0, pad_size))
        lam_base = torch.nn.functional.pad(lam_base, (0, pad_size), value=1e-6)

    if is_conv:
        # Match TLoraModule._initialize_from_svd: only keep the first
        # ``in_dim`` columns of the right-singular vectors, matching the
        # 1x1 conv layout stored in q_layer.weight.
        q_base = q_base[:, :in_dim].contiguous()

    lam_base = lam_base.unsqueeze(0)  # (1, lora_dim)
    return q_base, p_base, lam_base


def convert(
    tlora_sd: Dict[str, torch.Tensor],
    base_sd: Dict[str, torch.Tensor],
    sig_type: str = "principal",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, torch.Tensor]:
    """Convert a legacy T-LoRA state dict into standard LoRA format."""
    dev = torch.device(device)
    groups = _group_tlora_keys(tlora_sd)
    if not groups:
        raise ValueError(
            "No legacy T-LoRA keys (q_layer / p_layer / lambda_layer) found "
            "in the input checkpoint.  Nothing to convert."
        )

    wanted_lora_names = set(groups.keys())
    base_lookup = _build_base_lookup(base_sd, wanted_lora_names)

    if not base_lookup:
        sample_base = list(base_sd.keys())[:5]
        sample_lora = list(wanted_lora_names)[:5]
        raise RuntimeError(
            "Could not match any T-LoRA module to a weight in the base "
            "model.\n"
            f"  Sample T-LoRA names: {sample_lora}\n"
            f"  Sample base keys:    {sample_base}\n"
            "Make sure --base-model points to the underlying anima "
            "checkpoint, not another LoRA file."
        )

    converted: Dict[str, torch.Tensor] = {}
    missing_base: list[str] = []
    skipped: list[str] = []

    for lora_name, parts in sorted(groups.items()):
        q = parts.get("q_layer.weight")
        p = parts.get("p_layer.weight")
        lam = parts.get("lambda_layer")
        alpha = parts.get("alpha")
        if q is None or p is None or lam is None:
            skipped.append(lora_name)
            continue

        base_key = base_lookup.get(lora_name)
        if base_key is None:
            missing_base.append(lora_name)
            continue

        base_weight = base_sd[base_key]
        lora_dim = q.shape[0]

        q_base, p_base, lam_base = _svd_base(
            base_weight, lora_dim, sig_type, dev
        )

        q_t = q.detach().to(device=dev, dtype=torch.float32)
        p_t = p.detach().to(device=dev, dtype=torch.float32)
        lam_t = lam.detach().to(device=dev, dtype=torch.float32)
        if lam_t.dim() == 1:
            lam_t = lam_t.unsqueeze(0)

        is_conv = q_t.dim() > 2
        if is_conv:
            out_ch = p_t.shape[0]
            in_ch = q_t.shape[1]
            q_2d = q_t.reshape(lora_dim, -1)
            p_2d = p_t.reshape(out_ch, lora_dim)
            q_base_2d = q_base  # already 2D
            p_base_2d = p_base  # already 2D

            p_scaled = p_2d * lam_t
            p_base_scaled = p_base_2d * lam_base

            lora_up_2d = torch.cat([p_scaled, -p_base_scaled], dim=1)
            lora_down_2d = torch.cat([q_2d, q_base_2d], dim=0)

            extra_dims = q_t.dim() - 2
            kernel_ones = [1] * extra_dims
            lora_up = lora_up_2d.reshape(
                out_ch, 2 * lora_dim, *kernel_ones
            ).contiguous()
            lora_down = lora_down_2d.reshape(
                2 * lora_dim, in_ch, *kernel_ones
            ).contiguous()
        else:
            p_scaled = p_t * lam_t
            p_base_scaled = p_base * lam_base

            lora_up = torch.cat([p_scaled, -p_base_scaled], dim=1).contiguous()
            lora_down = torch.cat([q_t, q_base], dim=0).contiguous()

        if alpha is None:
            alpha_val = torch.tensor(float(lora_dim))
        else:
            alpha_val = alpha.detach().float().cpu()
        new_alpha = alpha_val * 2.0

        target_dtype = dtype if dtype is not None else q.dtype
        converted[f"{lora_name}.lora_up.weight"] = lora_up.to(
            dtype=target_dtype, device="cpu"
        )
        converted[f"{lora_name}.lora_down.weight"] = lora_down.to(
            dtype=target_dtype, device="cpu"
        )
        converted[f"{lora_name}.alpha"] = new_alpha.to(
            dtype=torch.float32, device="cpu"
        )

    # Pass through any non-TLoRA keys unchanged (e.g. metadata tensors).
    legacy_full_keys = set()
    for lora_name in groups:
        for suffix in ("q_layer.weight", "p_layer.weight", "lambda_layer", "alpha"):
            legacy_full_keys.add(f"{lora_name}.{suffix}")
    for k, v in tlora_sd.items():
        if k not in legacy_full_keys and k not in converted:
            converted[k] = v

    if missing_base:
        sample_base = [k for k in list(base_sd.keys())[:5] if k.endswith(".weight")]
        print(
            f"[warn] {len(missing_base)} T-LoRA modules had no matching base "
            f"weight and were skipped.\n"
            f"        first unmatched lora names: {missing_base[:5]}\n"
            f"        sample base weight keys:    {sample_base}",
            file=sys.stderr,
        )
    if skipped:
        print(
            f"[warn] {len(skipped)} T-LoRA modules had incomplete parts and "
            f"were skipped. First few: {skipped[:5]}",
            file=sys.stderr,
        )

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a legacy T-LoRA checkpoint into a standard LoRA "
            "checkpoint that ComfyUI can load."
        )
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Path to the base model state dict (e.g. anima .safetensors).",
    )
    parser.add_argument(
        "--tlora",
        required=True,
        help="Path to the legacy T-LoRA checkpoint to convert.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination path for the converted standard-LoRA checkpoint.",
    )
    parser.add_argument(
        "--sig-type",
        choices=["principal", "last", "middle"],
        default="principal",
        help=(
            "Which singular vectors the original T-LoRA run used for base "
            "initialization. Use the same value you trained with."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for the SVD reconstruction (cpu or cuda).",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Precision of the saved LoRA weights.",
    )
    args = parser.parse_args()

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    print(f"Loading base model:   {args.base_model}")
    base_sd = _load_state_dict(args.base_model)
    print(f"Loading T-LoRA:       {args.tlora}")
    tlora_sd = _load_state_dict(args.tlora)

    print("Converting ...")
    converted = convert(
        tlora_sd,
        base_sd,
        sig_type=args.sig_type,
        device=args.device,
        dtype=dtype_map[args.dtype],
    )

    print(f"Saving output to:     {args.output}")
    _save_state_dict(converted, args.output)
    print(
        f"Done. Wrote {sum(1 for k in converted if k.endswith('lora_up.weight'))} "
        "LoRA modules."
    )


if __name__ == "__main__":
    main()
