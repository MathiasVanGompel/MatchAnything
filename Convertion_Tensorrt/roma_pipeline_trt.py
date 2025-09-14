import torch
import torch.nn as nn

class ROMAPipelineTRT(nn.Module):
    """
    Thin ONNX-exportable wrapper around a RoMa 'outdoor' model that takes
    *tensors* and returns (warp, certainty) like roma_model.match(...).

    NOTE: You must connect the two TODOs to your embedded RoMa functions
    that:
      (1) build pyramid features from RGB tensors
      (2) run the regression matcher / decoder to produce (warp, certainty)
    See public RoMa for reference.  # https://github.com/Parskatt/RoMa
    """
    def __init__(self, roma_model, H: int, W: int):
        super().__init__()
        self.roma = roma_model
        self.H = H
        self.W = W

    @torch.no_grad()
    def forward(self, image0: torch.Tensor, image1: torch.Tensor):
        """
        image*: [B, 3, H, W] in [0,1]; returns:
          warp:      [B, H, 2*W, 2]  (concat A->B and B->A flow in RoMa demos)
          certainty: [B, 1, H, 2*W]  (or [B, H, 2*W] depending on your fork)
        """
        # TODO(1): compute multi-scale features from tensors (no file I/O).
        # In many forks this is something like:
        #   feats0 = self.roma.encoder(image0)   # dict with coarse/fine
        #   feats1 = self.roma.encoder(image1)
        # Replace with the correct call in your vendored codebase.
        feats0 = self.roma.encoder_from_tensor(image0)
        feats1 = self.roma.encoder_from_tensor(image1)

        # TODO(2): run the match decoder / regression head
        #   warp, certainty = self.roma.matcher.decode_from_feats(feats0, feats1)
        warp, certainty = self.roma.decode_from_feats(feats0, feats1)

        return warp, certainty
