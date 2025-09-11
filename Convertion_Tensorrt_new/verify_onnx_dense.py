#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np, cv2

def ceil_mul14(x): return ((x + 13)//14)*14

def load_rgb(path, size=None):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if size is not None:
        s = ceil_mul14(size)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x = np.transpose(rgb[None, ...], (0,3,1,2)).astype(np.float32)
    return x, img

sess = ort.InferenceSession("out/matchanything_dense.onnx", providers=["CPUExecutionProvider"])
in0, in1 = sess.get_inputs()[0].name, sess.get_inputs()[1].name
outW, outC = sess.get_outputs()[0].name, sess.get_outputs()[1].name

x0, im0 = load_rgb("IMG_A.jpg", size=840)
x1, im1 = load_rgb("IMG_B.jpg", size=840)
warp_c, cert_c = sess.run([outW, outC], {in0:x0, in1:x1})
print("warp_c:", warp_c.shape, "cert_c:", cert_c.shape)
