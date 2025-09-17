# TensorRT Conversion Notes

## Directory layout

- `full/` contains the working TensorRT prototype. It focuses on the encoder + GP head path and successfully loads 342 of the remapped weights while the pretrained checkpoint exposes 603 keys. The unified loader handles the gap by skipping mismatched tensors and reporting what is missing.
- `plus/` is an experimental playground that wires in decoder and refinement heads. These scripts do not yet restore the weights correctly, so the exported networks are placeholders for future research rather than ready-to-run models.
- `requirements_improved.txt` lists optional Python packages that make the conversion workflow smoother.

## Typical workflow

1. Export ONNX from the RoMa checkpoint (note the 518 resolution):

   ```bash
   python Convertion_Tensorrt/full/convert_full_matchanything_from_ckpt.py \
     --ckpt imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt \
     --onnx Convertion_Tensorrt/out/matchanything_full_518.onnx \
     --H 518 --W 518 --precision fp16 --opset 17
   ```

2. Run inference with the TensorRT engine built from that ONNX graph:

   ```bash
   python Convertion_Tensorrt/full/run_full_matchanything_trt.py \
     --engine Convertion_Tensorrt/out/matchanything_full_518_fp16.plan \
     --image0 tests/data/02928139_3448003521.jpg \
     --image1 tests/data/17295357_9106075285.jpg \
     --opt 518 --conf 0.25 --sample 0.0 --topk 5000 \
     --mutual --ransac --model H --rth 2.0 --viz cv2 --draw lines
   ```

### Why 518?

RoMa’s ViT-L/14 backbone expects image sizes that are multiples of the 14-pixel patch size. A resolution of 518 pixels corresponds to 37 patches on each axis (37 × 14 = 518), yielding the 37 × 37 + CLS token layout that the pretrained RoMa checkpoint was tuned for. Sticking to 518 keeps positional embeddings aligned and prevents costly resizing of attention tokens during export.

### Relationship to the upstream MatchAnything project

This fork diverges from the Hugging Face Space (`LittleFrog/MatchAnything`) and the main branch by patching files deep inside `MatchAnything/imcui/…`. The vendorized RoMa code and weight loading utilities under `imcui/third_party/MatchAnything` were expanded to remap checkpoints, resize positional embeddings, and back-fill missing DINOv2 parameters from TIMM. Those changes eliminate the namespace mismatches that otherwise block TensorRT export.

## Current status

- The `full/` pipeline is the reliable path for producing TensorRT engines today.
- The `plus/` directory captures aspirational work to integrate the RoMa decoder and refinement stages; once the weight mapping issues are solved it will be ready for experimentation.
- Keep an eye on loader logs: they document how many of the 342 usable weights were restored relative to the 603 keys shipped in the checkpoint and flag any further drift from upstream training code.
