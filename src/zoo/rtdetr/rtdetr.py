"""by lyuwenyu"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from src.core import register

__all__ = [
    "RTDETR",
]


@register
class RTDETR(nn.Module):
    __inject__ = ["backbone", "encoder", "decoder", "affordance_branch"]
    __share__ = ["use_affordance"]

    def __init__(
        self,
        backbone: nn.Module,
        encoder,
        decoder,
        affordance_branch,
        multi_scale=None,
        use_affordance=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.use_affordance = use_affordance
        self.affordance_branch = affordance_branch if use_affordance else None
        self.multi_scale = multi_scale

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        # Backbone
        x = self.backbone(x)

        # If the backbone returns more levels than the encoder consumes (e.g.
        # return_idx [0,1,2,3] adds C2 / stride-4), the extra finest level(s)
        # are routed to the affordance branch only — the detector (encoder +
        # decoder) is left untouched. The encoder asserts an exact level count.
        low_level_feat = None
        n_enc = len(self.encoder.in_channels)
        if isinstance(x, (list, tuple)) and len(x) > n_enc:  # noqa: UP038  (py3.8: X|Y breaks)
            low_level_feat = x[0]
            x = list(x[-n_enc:])

        # Encoder
        encoder_output = self.encoder(x)

        # Decoder
        decoder_output = self.decoder(encoder_output, targets)

        # # Debugging printing decoder output dimensions
        # print("Decoder output dimensions:")
        # for key, value in decoder_output.items():
        #     if isinstance(value, torch.Tensor):
        #         print(f"{key}: {value.shape}")
        #     elif isinstance(value, list):
        #         print(f"{key}: list of length {len(value)}")

        # Initialize output dictionary with detection results
        output = {
            "pred_logits": decoder_output["pred_logits"],
            "pred_boxes": decoder_output["pred_boxes"],
        }

        # Add affordance predictions only if the branch is enabled.
        # The head returns a shared feature map + per-query dynamic kernels
        # (no dense [Q, C, H, W] tensor); masks are decoded lazily for
        # matched/top-K queries by the criterion / postprocessor.
        if self.use_affordance and self.affordance_branch is not None:
            aff_out = self.affordance_branch(
                decoder_output["features"],
                encoder_output,
                low_level_feat=low_level_feat,
            )
            output["aff_feat"] = aff_out["aff_feat"]
            output["aff_kernel"] = aff_out["aff_kernel"]
            output["aff_meta"] = aff_out["aff_meta"]
            # 'dynamic' (default, AffordanceBranch) or 'embedding'
            # (AffordanceEmbeddingBranch, the A5 ablation alternative).
            output["aff_head_type"] = aff_out.get("aff_head_type", "dynamic")
            output["aff_box_relative"] = aff_out.get("aff_box_relative", False)
            # Dense auxiliary head (AffordanceBranch.dense_aux): top-level
            # output only — the aux/dn loss loops must NOT re-supervise it.
            if "aff_dense_logits" in aff_out:
                output["aff_dense_logits"] = aff_out["aff_dense_logits"]

            # Include auxiliary outputs if present
            if "aux_outputs" in decoder_output:
                # Deep affordance supervision: per-layer kernels for the
                # intermediate decoder layers (training only; None otherwise).
                # Shape [B, L-1, Q, n_params], aligned with the leading decoder
                # aux entries. The trailing encoder-top-k aux entry has no
                # decoder query embedding and is left detection-only.
                aux_aff_kernel = aff_out.get("aux_aff_kernel")
                output["aux_outputs"] = []
                for j, aux_out in enumerate(decoder_output["aux_outputs"]):
                    aux_output = {
                        "pred_logits": aux_out["pred_logits"],
                        "pred_boxes": aux_out["pred_boxes"],
                    }
                    if aux_aff_kernel is not None and j < aux_aff_kernel.shape[1]:
                        # Share the SAME feature map (not a clone) so every
                        # layer's affordance loss trains the one pixel branch.
                        aux_output["aff_feat"] = output["aff_feat"]
                        aux_output["aff_kernel"] = aux_aff_kernel[:, j]
                        aux_output["aff_meta"] = output["aff_meta"]
                        aux_output["aff_head_type"] = output["aff_head_type"]
                        aux_output["aff_box_relative"] = output["aff_box_relative"]
                    output["aux_outputs"].append(aux_output)

            # Include denoising outputs if present
            if "dn_aux_outputs" in decoder_output:
                output["dn_aux_outputs"] = []
                for dn_aux_out in decoder_output["dn_aux_outputs"]:
                    dn_aux_output = {
                        "pred_logits": dn_aux_out["pred_logits"],
                        "pred_boxes": dn_aux_out["pred_boxes"],
                    }
                    output["dn_aux_outputs"].append(dn_aux_output)
        else:
            # Include auxiliary outputs without affordance predictions
            if "aux_outputs" in decoder_output:
                output["aux_outputs"] = []
                for aux_out in decoder_output["aux_outputs"]:
                    output["aux_outputs"].append(
                        {
                            "pred_logits": aux_out["pred_logits"],
                            "pred_boxes": aux_out["pred_boxes"],
                        }
                    )

            # Include denoising outputs without affordance predictions
            if "dn_aux_outputs" in decoder_output:
                output["dn_aux_outputs"] = []
                for dn_aux_out in decoder_output["dn_aux_outputs"]:
                    output["dn_aux_outputs"].append(
                        {
                            "pred_logits": dn_aux_out["pred_logits"],
                            "pred_boxes": dn_aux_out["pred_boxes"],
                        }
                    )

        # One-to-many auxiliary group (detection-only; passes through in both
        # the affordance and non-affordance paths). Absent at inference.
        if "o2m_aux_outputs" in decoder_output:
            output["o2m_aux_outputs"] = [
                {"pred_logits": o["pred_logits"], "pred_boxes": o["pred_boxes"]}
                for o in decoder_output["o2m_aux_outputs"]
            ]
        if "o2m_meta" in decoder_output:
            output["o2m_meta"] = decoder_output["o2m_meta"]

        if "dn_meta" in decoder_output:
            output["dn_meta"] = decoder_output["dn_meta"]

        return output

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self
