# TensorRT Conversion Notes

## Directory layout

- `full/` contains the working TensorRT prototype. It focuses on the encoder + GP head path and successfully loads 342 of the remapped weights while the pretrained checkpoint exposes 603 keys. The unified loader handles the gap by skipping mismatched tensors and reporting what is missing.
- `plus/` is an newer version that wires in decoder and refinement heads. These scripts do not yet restore the weights correctly, so this is just to give an idea on how to advance the TRT engine.
- `EloFTR/` is an older version that translates the EloFTr into TensorRT. I have not touched this in a while, but normally everything still works. This gave quite good results from early on and gave way less errors for ONNX export than RoMa.
- `requirements_improved.txt` lists packages.

## Typical workflow
Here is an example of how the engine is loaded and how images are runned through the engine (for files in `full/`).
1. Export ONNX from the RoMa checkpoint (note the 518 resolution):

   ```bash
   python Conversion_Tensorrt/full/convert_full_matchanything_from_ckpt.py \
     --ckpt imcui/third_party/MatchAnything/weights/matchanything_roma.ckpt \
     --onnx Conversion_Tensorrt/out/matchanything_full_518.onnx \
     --H 518 --W 518 --precision fp16 --opset 17
   ```
2. 

  ```bash
  /usr/src/tensorrt/bin/trtexec \ 
  --onnx=Conversion_Tensorrt/out/matchanything_full_518.onnx \ 
  --saveEngine=Conversion_Tensorrt/out/matchanything_full_518_fp16.plan \ 
  --minShapes=image0:1x3x518x518,image1:1x3x518x518 \ 
  --optShapes=image0:1x3x518x518,image1:1x3x518x518 \ 
  --maxShapes=image0:1x3x518x518,image1:1x3x518x518 \ 
  --fp16 --builderOptimizationLevel=3 --memPoolSize=workspace:1024
  ```
3. Run inference with the TensorRT engine built from that ONNX graph:

   ```bash
   python Conversion_Tensorrt/full/run_full_matchanything_trt.py \
     --engine Conversion_Tensorrt/out/matchanything_full_518_fp16.plan \
     --image0 tests/data/02928139_3448003521.jpg \
     --image1 tests/data/17295357_9106075285.jpg \
     --opt 518 --conf 0.25 --sample 0.0 --topk 5000 \
     --mutual --ransac --model H --rth 2.0 --viz cv2 --draw lines
   ```

### Why 518?
RoMa’s ViT-L/14 backbone expects image sizes that are multiples of the 14-pixel patch size. A resolution of 518 pixels corresponds to 37 patches on each axis (37 × 14 = 518), yielding the 37 × 37 + CLS token layout that the pretrained RoMa checkpoint was tuned for. On stronger GPU, this can be changed to have a bigger resolution, but it needs to be a multiple of 14.

### Relationship to the original MatchAnything project

Apart from all the files in Conversion_Tensorrt, there have been made small changes to files to make it ONNX friendly (I do not think all  these changes actually matter anymore as i do not use all these function in conversion, (was for previous prototype) but are included here for information). These changes are in:
- `imcui/hloc/extract_features.py`
- `imcui/hloc/match_dense.py`
- `imcui/hloc/matchers/duster.py`
- `imcui/third_party/MatchAnything/src/loftr/utils/geometry.py`
- `imcui/third_party/MatchAnything/src/utils/dataset.py`
- `imcui/third_party/MatchAnything/third_party/ROMA/roma/models/croco/pos_embed.py`
- `imcui/third_party/MatchAnything/third_party/ROMA/roma/models/dust3r/utils/geometry.py`
- `imcui/third_party/MatchAnything/third_party/ROMA/roma/models/dust3r/utils/image.py`
- `imcui/third_party/MatchAnything/third_party/ROMA/roma/models/matcher.py`
- `imcui/third_party/MatchAnything/third_party/ROMA/roma/models/transformer/dinov2.py`
- `imcui/third_party/MatchAnything/third_party/ROMA/roma/utils/utils.py`

## Current status

- The `full/` pipeline is the reliable path for producing TensorRT engines.
- The `plus/` directory captures aspirational work to integrate the RoMa decoder and refinement stages; once the weight mapping issues are solved it will be ready for experimentation.
- Keep an eye on loader logs: they document how many of the 342 usable weights were restored relative to the 603 keys shipped in the checkpoint and flag any further drift from upstream training code. In the `plus/` directory I tried to load more of the checkpoints, but it does not work yet.
