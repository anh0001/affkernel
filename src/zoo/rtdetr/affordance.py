"""
AffKernel: Dynamic-Kernel Per-Instance Affordance Segmentation Head for RT-DETR
==============================================================================

This replaces the original dense ``AffordanceBranch`` (which materialised a
``[batch, num_queries, num_classes, 244, 244]`` tensor for *every* query and
decoded masks from a 49-dim query projection with no access to image features).
That design caused the OOM / slowness and could not localise affordances.

Design (CondInst / MaskDINO style):

1. A *single* shared affordance-feature map ``F`` is built once per image from
   the finest HybridEncoder feature (stride 8), via a light conv stack
   (no pixel self-attention). Channels = ``reduced_dim``. Two normalised
   coordinate channels are appended (CoordConv) to give the dynamic kernels
   spatial grounding.
2. Each RT-DETR decoder query predicts the *parameters of a tiny 2-layer
   1x1 dynamic conv* (a few hundred floats), not a dense mask.
3. The per-instance affordance mask is produced only for the queries that
   matter (Hungarian-matched at train time, top-K at inference) by applying
   that query's dynamic conv to ``F``. The dense ``[Q, C, H, W]`` tensor is
   never created.

Forward returns a dict so the criterion / postprocessor can decode masks
lazily for selected queries:

    {
        'aff_feat':   [batch, reduced_dim, Hf, Wf]   # shared, per image
        'aff_kernel': [batch, num_queries, n_params]  # per query
        'aff_meta':   (reduced_dim, Hf, Wf)           # decode metadata
    }

Use :func:`decode_affordance_masks` to turn selected kernels into masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core import register

__all__ = [
    "AffordanceBranch",
    "AffordanceEmbeddingBranch",
    "MSReadout",
    "decode_affordance_masks",
    "decode_affordance_masks_embedding",
]

# Dynamic head topology: input (reduced_dim + 2 coord) -> DYN_HIDDEN -> output_dim.
# Widened 8 -> 16: the narrow head was a capacity bottleneck for a 10-way
# per-instance affordance decode.
DYN_HIDDEN = 16

# Number of per-instance box-relative geometry channels appended at decode time
# when ``use_box_relative`` is on: (rx, ry, inside-box indicator). See
# ``_box_relative_channels``.
N_BOX_REL_CHANNELS = 3
# Clamp magnitude for the box-relative offset channels. Tiny boxes would
# otherwise produce huge offsets far outside the box; clamping keeps the
# dynamic kernel's input bounded.
_REL_CLAMP = 4.0


def _dynamic_layer_sizes(in_ch: int, hidden: int, out_ch: int):
    """Weight/bias param counts for the 2-layer 1x1 dynamic conv."""
    weights = [in_ch * hidden, hidden * out_ch]
    biases = [hidden, out_ch]
    return weights, biases


def _box_relative_channels(boxes: torch.Tensor, h: int, w: int, device, dtype) -> torch.Tensor:
    """Per-instance object-relative geometry channels, shape [N, 3, H, W].

    The shared affordance feature map only carries *absolute* image position
    (CoordConv). Affordances are object-relative (a handle is the right edge of
    *this* object's box; a cutting edge, a pound face, a support surface are all
    defined relative to the instance). For each selected query's box
    (cx, cy, bw, bh) in normalised [0, 1] coords we give the dynamic kernel:

      * ``rx = (x - cx) / (bw / 2)`` — signed horizontal offset, in box-half-widths
      * ``ry = (y - cy) / (bh / 2)`` — signed vertical offset, in box-half-heights
      * ``inside`` — 1.0 where ``|rx| <= 1 and |ry| <= 1`` (inside the box), else 0.0

    rx/ry are clamped to ``[-_REL_CLAMP, _REL_CLAMP]`` so tiny boxes stay bounded.
    """
    n = boxes.shape[0]
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")  # [H, W] each, in [0, 1]
    cx = boxes[:, 0].view(n, 1, 1)
    cy = boxes[:, 1].view(n, 1, 1)
    bw = boxes[:, 2].view(n, 1, 1).clamp(min=1e-3)
    bh = boxes[:, 3].view(n, 1, 1).clamp(min=1e-3)
    rx = ((gx.unsqueeze(0) - cx) / (bw / 2)).clamp(-_REL_CLAMP, _REL_CLAMP)
    ry = ((gy.unsqueeze(0) - cy) / (bh / 2)).clamp(-_REL_CLAMP, _REL_CLAMP)
    inside = ((rx.abs() <= 1.0) & (ry.abs() <= 1.0)).to(dtype)
    return torch.stack([rx, ry, inside], dim=1)  # [N, 3, H, W]


class MSReadout(nn.Module):
    """Multi-scale FPN mask-feature generator (MS-lite lever).

    Replaces the single-level ``mask_feat`` (stride-8 encoder -> learned 2x
    upsamples + C2 lateral) with a top-down FPN over three levels — C2
    (stride-4, backbone), the stride-8 and stride-16 encoder maps — fused to
    stride-4, then a single learned 2x upsample to the SAME stride-2 decode
    grid the stride-2 head already uses. Resolution is held constant so the
    lever isolates feature richness (the diagnosis: missed regions are
    representation blind spots), not decode resolution.

    Exposes ``self.proj`` (1x1 penult->reduced_dim) as the final layer so a
    frozen linear probe / hook can read the ``penult_dim``-channel penultimate
    feature (pre-registered S3-lite tap point).
    """

    def __init__(
        self,
        c2_dim: int,
        enc_dim: int,
        reduced_dim: int,
        fpn_dim: int = 64,
        penult_dim: int = 128,
    ):
        super().__init__()
        gn = min(8, fpn_dim)
        self.lat_c2 = nn.Conv2d(c2_dim, fpn_dim, kernel_size=1)  # stride-4
        self.lat_s8 = nn.Conv2d(enc_dim, fpn_dim, kernel_size=1)  # stride-8
        self.lat_s16 = nn.Conv2d(enc_dim, fpn_dim, kernel_size=1)  # stride-16

        def smooth():
            return nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),
                nn.GroupNorm(gn, fpn_dim),
                nn.SiLU(inplace=True),
            )

        self.smooth_s8 = smooth()
        self.smooth_c2 = smooth()
        # stride-4 (P2) -> stride-2 penult, then 3x3 refine.
        pgn = min(8, penult_dim)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(fpn_dim, penult_dim, kernel_size=2, stride=2),
            nn.GroupNorm(pgn, penult_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(penult_dim, penult_dim, kernel_size=3, padding=1),
            nn.GroupNorm(pgn, penult_dim),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Conv2d(penult_dim, reduced_dim, kernel_size=1)

    def forward(self, c2, s8, s16):
        p4 = self.lat_s16(s16)
        p3 = self.smooth_s8(self.lat_s8(s8) + F.interpolate(p4, size=s8.shape[-2:], mode="nearest"))
        p2 = self.smooth_c2(self.lat_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest"))
        return self.proj(self.up(p2))  # [B, reduced_dim, 2*Hc2, 2*Wc2] (stride-2)


@register
class AffordanceBranch(nn.Module):
    """Dynamic-kernel affordance head.

    Args:
        input_dim: number of decoder layers produced by RTDETRTransformer
            (kept for config compatibility; only the last layer is used).
        hidden_dim: decoder feature dimension (query embedding size).
        output_dim: number of affordance classes including background.
        reduced_dim: channel width of the shared affordance-feature map.
        mask_upsample_factor: total learned upsample of ``mask_feat`` (power of
            2), applied only when ``use_mask_upsample`` is True. 2 -> stride-4
            (default); 4 -> stride-2 (~320x320), finer decode + supervision grid.
        low_level_dim: channel count of the backbone C2 (stride-4) feature.
            When set (>0), a lateral projects C2 and is added to the shared
            affordance map, giving the kernels *real* stride-4 detail instead
            of a learned 2x upsample of the stride-8 encoder feature. ``None``
            (default) keeps the v2 head (no lateral) — the detector is never
            touched either way.
        use_box_relative: when True, append per-instance object-relative
            geometry channels (rx, ry, inside) to the dynamic-kernel input at
            decode time (see :func:`_box_relative_channels`). Grows ``dyn_in``
            by :data:`N_BOX_REL_CHANNELS`.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        reduced_dim: int = 8,
        use_mask_upsample: bool = True,
        mask_upsample_factor: int = 2,
        dyn_hidden: int = DYN_HIDDEN,
        low_level_dim: int | None = None,
        use_box_relative: bool = False,
        deep_supervision: bool = False,
        dense_aux: bool = False,
        ms_readout: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reduced_dim = reduced_dim
        self.use_mask_upsample = use_mask_upsample
        self.mask_upsample_factor = mask_upsample_factor
        self.dyn_hidden = dyn_hidden
        self.low_level_dim = low_level_dim
        self.use_box_relative = use_box_relative
        # When True, also generate per-query kernels for the intermediate
        # decoder layers (training only) so the criterion can supervise the
        # affordance head at every layer (deep supervision). The shared
        # `kernel_head` is reused on each layer's query embeddings, so this
        # adds NO parameters and the eval/inference path is byte-identical
        # (only the last decoder layer is decoded) => zero inference cost.
        self.deep_supervision = deep_supervision

        # MS-lite multi-scale FPN readout (mutually exclusive with the
        # single-level mask_feat below). Fuses C2 + stride-8 + stride-16 into a
        # stride-2 aff_feat. Everything downstream (dynamic kernels, deep
        # supervision, full-res loss, postproc) is unchanged — only the feature
        # GENERATOR is swapped.
        self.ms_readout = None
        if ms_readout:
            if not low_level_dim:
                raise ValueError("ms_readout requires low_level_dim (backbone C2)")
            self.ms_readout = MSReadout(
                c2_dim=low_level_dim, enc_dim=hidden_dim, reduced_dim=reduced_dim
            )

        # Shared affordance-feature map. The finest encoder level is stride 8
        # (~80x80 @ 640 input). With `use_mask_upsample=True` (v2 default), a
        # learned 2x upsample (ConvTranspose) takes it to stride 4 (~160x160).
        # The boundary-weighted F_beta^w metric is starved by coarse masks, so
        # the extra resolution is the single biggest accuracy lever here.
        # `mask_upsample_factor` controls the TOTAL upsample of `mask_feat`
        # (must be a power of 2): 2 -> stride-4 (v2/v3 default, one ConvTranspose);
        # 4 -> stride-2 (~320x320, two ConvTranspose stages) which gives both the
        # decode AND the supervision grid finer detail for thin classes.
        # `use_mask_upsample=False` is the LOO ablation that reverts to the
        # pre-v2 stride-8 head. The C2 lateral and decode/loss/postproc all read
        # spatial dims off aff_feat.shape, so they adapt automatically.
        # Skipped entirely under MS-lite (MSReadout replaces this generator).
        self.mask_feat = None
        self.low_lateral = None
        mask_layers = [
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.ReLU(inplace=True),
        ]
        n_upsample = 0
        if not ms_readout and use_mask_upsample:
            if mask_upsample_factor < 1 or (mask_upsample_factor & (mask_upsample_factor - 1)) != 0:
                raise ValueError(
                    "mask_upsample_factor must be a power of 2, got " f"{mask_upsample_factor}"
                )
            n_upsample = mask_upsample_factor.bit_length() - 1
        for _ in range(n_upsample):
            mask_layers += [
                nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 2, kernel_size=2, stride=2),
                nn.GroupNorm(8, hidden_dim // 2),
                nn.ReLU(inplace=True),
            ]
        mask_layers.append(nn.Conv2d(hidden_dim // 2, reduced_dim, kernel_size=3, padding=1))
        if not ms_readout:
            self.mask_feat = nn.Sequential(*mask_layers)

        # Optional C2 (stride-4) lateral. Kept as a SEPARATE additive module so
        # `mask_feat`'s parameter layout is byte-identical to v2 (v2 checkpoints
        # still load). Projects backbone C2 to reduced_dim and adds it to the
        # shared affordance map at stride-4 resolution.
        if not ms_readout and low_level_dim:
            gn_groups = min(8, reduced_dim)
            self.low_lateral = nn.Sequential(
                nn.Conv2d(low_level_dim, reduced_dim, kernel_size=1),
                nn.GroupNorm(gn_groups, reduced_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced_dim, reduced_dim, kernel_size=3, padding=1),
            )

        # Per-query dynamic-kernel parameter generator.
        dyn_in = reduced_dim + 2  # + 2 CoordConv channels
        if use_box_relative:
            dyn_in += N_BOX_REL_CHANNELS  # + per-instance (rx, ry, inside)
        self._w_sizes, self._b_sizes = _dynamic_layer_sizes(dyn_in, self.dyn_hidden, output_dim)
        n_params = sum(self._w_sizes) + sum(self._b_sizes)
        self.kernel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_params),
        )
        self.dyn_in = dyn_in

        # Optional jointly-trained dense (box-independent) affordance head:
        # a 1x1 classifier on the shared aff_feat (F recipe — measured NEGATIVE,
        # kept for the ablation record). Trained with an image-level semantic
        # loss (SetCriterion.loss_affordances_dense); emitted in train AND eval.
        self.dense_head = None
        if dense_aux:
            self.dense_head = nn.Conv2d(reduced_dim, output_dim, kernel_size=1)

    @staticmethod
    def _coord_channels(feat: torch.Tensor) -> torch.Tensor:
        """Two normalised [-1, 1] coordinate channels, shape [B, 2, H, W]."""
        b, _, h, w = feat.shape
        ys = torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype)
        xs = torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx, gy], dim=0)  # [2, H, W]
        return coords.unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, decoder_features: torch.Tensor, encoder_feats, low_level_feat=None) -> dict:
        """
        Args:
            decoder_features: [batch, num_layers, num_queries, hidden_dim].
            encoder_feats: list of HybridEncoder feature maps; index 0 is the
                finest (stride 8) [batch, hidden_dim, Hf, Wf].
            low_level_feat: optional backbone C2 (stride 4) feature
                [batch, low_level_dim, Hc, Wc], used only when this branch was
                built with ``low_level_dim``.

        Returns:
            dict with 'aff_feat', 'aff_kernel', 'aff_meta', 'aff_box_relative'.
        """
        if decoder_features.dim() == 3:
            decoder_features = decoder_features.unsqueeze(0)
        elif decoder_features.dim() != 4:
            raise ValueError("decoder_features must be a 3D or 4D tensor.")

        # Use only the last decoder layer's query embeddings.
        query_feat = decoder_features[:, -1]  # [B, Q, hidden_dim]

        if self.ms_readout is not None:
            # MS-lite: FPN over C2 (stride-4) + stride-8/16 encoder maps.
            if low_level_feat is None:
                raise ValueError("ms_readout requires low_level_feat (backbone C2)")
            aff_feat = self.ms_readout(
                low_level_feat, encoder_feats[0], encoder_feats[1]
            )  # [B, reduced_dim, Hf, Wf] at stride-2
        else:
            finest = encoder_feats[0]  # [B, hidden_dim, Hf, Wf]
            aff_feat = self.mask_feat(finest)  # [B, reduced_dim, Hf, Wf]

            # Add real stride-4 detail from backbone C2.
            if self.low_lateral is not None and low_level_feat is not None:
                lat = self.low_lateral(low_level_feat)  # [B, reduced_dim, Hc, Wc]
                if lat.shape[-2:] != aff_feat.shape[-2:]:
                    lat = F.interpolate(
                        lat,
                        size=aff_feat.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                aff_feat = aff_feat + lat

        kernels = self.kernel_head(query_feat)  # [B, Q, n_params]

        out = {
            "aff_feat": aff_feat,
            "aff_kernel": kernels,
            "aff_meta": (self.reduced_dim, aff_feat.shape[-2], aff_feat.shape[-1]),
            "aff_box_relative": self.use_box_relative,
        }

        if self.dense_head is not None:
            # Dense box-independent class logits [B, output_dim, Hf, Wf].
            out["aff_dense_logits"] = self.dense_head(aff_feat)

        # Deep supervision (training only): per-query kernels for the
        # intermediate decoder layers [0 .. L-2]. These align 1:1 with the
        # decoder aux_outputs entries (the trailing encoder-top-k aux entry is
        # excluded by the model wrapper, which only decorates the first L-1
        # entries). The eval path never enters this branch (`self.training` is
        # False), so inference stays byte-identical / zero added cost.
        if self.deep_supervision and self.training and decoder_features.shape[1] > 1:
            out["aux_aff_kernel"] = self.kernel_head(
                decoder_features[:, :-1]
            )  # [B, L-1, Q, n_params]

        return out


def _infer_dyn_hidden(n_params: int, dyn_in: int, output_dim: int) -> int:
    """Recover the dynamic-conv hidden width from the kernel parameter count.

    n_params = hidden*(dyn_in + output_dim + 1) + output_dim, see
    `_dynamic_layer_sizes`. Decoding stays config-free (no need to plumb
    `dyn_hidden` through the criterion/postprocessor) — useful for the LOO
    kernel-size ablation where 8 and 16 coexist across runs.
    """
    denom = dyn_in + output_dim + 1
    numer = n_params - output_dim
    if numer <= 0 or numer % denom != 0:
        raise ValueError(
            f"Cannot infer dyn_hidden from n_params={n_params}, dyn_in={dyn_in}, "
            f"output_dim={output_dim}"
        )
    return numer // denom


def _split_dynamic_params(params: torch.Tensor, dyn_in: int, output_dim: int, hidden: int):
    """Split a flat [N, n_params] tensor into per-instance conv weights/biases."""
    w_sizes, b_sizes = _dynamic_layer_sizes(dyn_in, hidden, output_dim)
    splits = w_sizes + b_sizes
    chunks = torch.split(params, splits, dim=1)
    n = params.shape[0]
    w1 = chunks[0].reshape(n, hidden, dyn_in, 1, 1)
    w2 = chunks[1].reshape(n, output_dim, hidden, 1, 1)
    b1 = chunks[2].reshape(n, hidden)
    b2 = chunks[3].reshape(n, output_dim)
    return (w1, b1), (w2, b2)


def decode_affordance_masks(
    aff_feat: torch.Tensor,
    kernels: torch.Tensor,
    batch_index: torch.Tensor,
    output_dim: int,
    boxes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode affordance logits for a *selected* set of queries only.

    Args:
        aff_feat: [B, reduced_dim, Hf, Wf] shared feature map.
        kernels: [N, n_params] dynamic params for the N selected queries.
        batch_index: [N] image index in the batch for each selected query.
        output_dim: number of affordance classes.
        boxes: optional [N, 4] cxcywh boxes (normalised [0, 1]) for the selected
            queries. When given, per-instance object-relative channels are
            appended to the dynamic-kernel input (see
            :func:`_box_relative_channels`); must match the ``use_box_relative``
            setting the kernels were generated for.

    Returns:
        [N, output_dim, Hf, Wf] affordance logits — never an [Q, C, H, W] tensor.
    """
    n = kernels.shape[0]
    if n == 0:
        return aff_feat.new_zeros((0, output_dim, aff_feat.shape[-2], aff_feat.shape[-1]))

    b, c, h, w = aff_feat.shape
    # Append CoordConv channels (recomputed cheaply, matches forward()).
    ys = torch.linspace(-1, 1, h, device=aff_feat.device, dtype=aff_feat.dtype)
    xs = torch.linspace(-1, 1, w, device=aff_feat.device, dtype=aff_feat.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx, gy], dim=0).unsqueeze(0).expand(b, -1, -1, -1)
    feat = torch.cat([aff_feat, coords], dim=1)  # [B, reduced_dim+2, H, W]

    selected = feat[batch_index]  # [N, reduced_dim+2, H, W]
    if boxes is not None:
        rel = _box_relative_channels(
            boxes.to(aff_feat.device, aff_feat.dtype),
            h,
            w,
            aff_feat.device,
            aff_feat.dtype,
        )  # [N, N_BOX_REL_CHANNELS, H, W]
        selected = torch.cat([selected, rel], dim=1)
    dyn_in = selected.shape[1]
    hidden = _infer_dyn_hidden(kernels.shape[1], dyn_in, output_dim)
    (w1, b1), (w2, b2) = _split_dynamic_params(kernels, dyn_in, output_dim, hidden)

    # Per-instance 1x1 conv implemented via grouped conv over N instances.
    x = selected.reshape(1, n * dyn_in, h, w)
    w1 = w1.reshape(n * hidden, dyn_in, 1, 1)
    x = F.conv2d(x, w1, b1.reshape(-1), groups=n)
    x = F.relu(x)
    w2 = w2.reshape(n * output_dim, hidden, 1, 1)
    x = F.conv2d(x, w2, b2.reshape(-1), groups=n)
    return x.reshape(n, output_dim, h, w)


@register
class AffordanceEmbeddingBranch(nn.Module):
    """MaskFormer-style mask-embedding alternative to the dynamic-kernel head.

    Per-query mask logits are produced by dot-product between a per-query,
    per-class embedding (predicted by a 2-layer MLP from the query feature)
    and a shared pixel-feature map (channels = `embed_dim`). The criterion +
    postprocessor dispatch on ``outputs['aff_head_type']`` (``'embedding'``
    here vs ``'dynamic'`` for :class:`AffordanceBranch`).

    Built as the A5 ablation refuting the anti-claim "the dynamic kernel is
    decoration; any head works": a natural, principled alternative mechanism
    held at a matched compute/parameter budget. Mirrors the dynamic-kernel
    head's stride/CoordConv/upsample choices to keep the comparison clean.

    The "kernel" field in the output is repurposed as the flat per-query
    embedding tensor of shape ``[B, Q, output_dim * embed_dim]`` so the
    existing aff_kernel plumbing in the model wrapper is reused unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        reduced_dim: int = 16,
        use_mask_upsample: bool = True,
        embed_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reduced_dim = reduced_dim
        self.use_mask_upsample = use_mask_upsample
        # The shared pixel-feature width and the per-class embedding length
        # must match for the dot product. Default to reduced_dim so the
        # parameter count is governed by the same lever as the dynamic head.
        self.embed_dim = embed_dim if embed_dim is not None else reduced_dim

        # Pixel branch — same architecture knob as AffordanceBranch so this
        # is a head-mechanism ablation, not a backbone/feature ablation.
        mask_layers = [
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.ReLU(inplace=True),
        ]
        if use_mask_upsample:
            mask_layers += [
                nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 2, kernel_size=2, stride=2),
                nn.GroupNorm(8, hidden_dim // 2),
                nn.ReLU(inplace=True),
            ]
        mask_layers.append(nn.Conv2d(hidden_dim // 2, self.embed_dim, kernel_size=3, padding=1))
        self.mask_feat = nn.Sequential(*mask_layers)

        # Per-query embedding head: produces output_dim x embed_dim per query.
        self.embed_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim * self.embed_dim),
        )

    def forward(self, decoder_features: torch.Tensor, encoder_feats, low_level_feat=None) -> dict:
        if decoder_features.dim() == 3:
            decoder_features = decoder_features.unsqueeze(0)
        elif decoder_features.dim() != 4:
            raise ValueError("decoder_features must be a 3D or 4D tensor.")

        query_feat = decoder_features[:, -1]  # [B, Q, hidden_dim]
        finest = encoder_feats[0]  # [B, hidden_dim, Hf, Wf]
        aff_feat = self.mask_feat(finest)  # [B, embed_dim, Hf, Wf]
        embeddings = self.embed_head(query_feat)  # [B, Q, C*embed_dim]

        return {
            "aff_feat": aff_feat,
            "aff_kernel": embeddings,  # flat [B,Q,C*d] kept under
            # the 'kernel' key to reuse
            # the model wrapper plumbing
            "aff_meta": (self.embed_dim, aff_feat.shape[-2], aff_feat.shape[-1]),
            "aff_head_type": "embedding",
            "aff_box_relative": False,
        }


def decode_affordance_masks_embedding(
    aff_feat: torch.Tensor,
    kernels: torch.Tensor,
    batch_index: torch.Tensor,
    output_dim: int,
    boxes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode embedding-head masks for a *selected* set of queries only.

    Logits[n, c, h, w] = sum_d  embedding[n, c, d] * pixel_feat[batch[n], d, h, w]

    ``boxes`` is accepted for a uniform call signature with
    :func:`decode_affordance_masks` and ignored (the embedding head has no
    box-relative path).
    """
    n = kernels.shape[0]
    if n == 0:
        return aff_feat.new_zeros((0, output_dim, aff_feat.shape[-2], aff_feat.shape[-1]))

    _b, c, _h, _w = aff_feat.shape
    embed_dim = kernels.shape[1] // output_dim
    if c != embed_dim:
        raise ValueError(
            f"embedding decode requires aff_feat channels (={c}) == "
            f"embed_dim (={embed_dim} inferred from kernels)."
        )
    emb = kernels.reshape(n, output_dim, embed_dim)  # [N, C, d]
    pix = aff_feat[batch_index]  # [N, d, H, W]
    # einsum: for each instance n, class c, pixel hw: sum_d emb[n,c,d]*pix[n,d,h,w]
    return torch.einsum("ncd,ndhw->nchw", emb, pix)
