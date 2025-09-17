# Visualization helpers mirroring the Hugging Face styling.
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch  # Documentation: https://matplotlib.org/stable/api/patches_api.html#matplotlib.patches.ConnectionPatch

# Match the green palette used on the Hugging Face Space (RGB for Matplotlib, BGR for OpenCV).
CV2_GREEN = (0, 180, 100)                     # BGR ordering for OpenCV.
MPL_GREEN = (0/255, 180/255, 100/255)         # RGB ordering for Matplotlib.

def draw_matches_cv2_green(im0, im1, kpts0, kpts1, line_thickness=2, dot_radius=3, margin=16):
    """Concatenate images and draw thin anti-aliased green lines + small dots (RGB in, RGB out)."""
    h = max(im0.shape[0], im1.shape[0])
    w0, w1 = im0.shape[1], im1.shape[1]
    canvas = np.ones((h, w0 + margin + w1, 3), dtype=np.uint8) * 255
    canvas[:im0.shape[0], :w0] = im0
    canvas[:im1.shape[0], w0 + margin:w0 + margin + w1] = im1

    offset = np.array([w0 + margin, 0.0], dtype=np.float32)
    for p0, p1 in zip(kpts0, kpts1):
        p0 = tuple(np.round(p0).astype(int))
        p1 = tuple(np.round(p1 + offset).astype(int))
        cv2.line(canvas, p0, p1, CV2_GREEN, int(line_thickness), lineType=cv2.LINE_AA)
        if dot_radius > 0:
            cv2.circle(canvas, p0, int(dot_radius), CV2_GREEN, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p1, int(dot_radius), CV2_GREEN, -1, lineType=cv2.LINE_AA)
    return canvas

def draw_matches_hf(im0, im1, kpts0, kpts1, line_width=2.0, dot_size=4.0, outpath=None, dpi=150, show=False):
    """HF-style two panels connected with green lines (uses ConnectionPatch across Axes)."""
    H0, W0 = im0.shape[:2]; H1, W1 = im1.shape[:2]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    ax0.imshow(im0); ax1.imshow(im1)
    for ax, (H, W) in zip((ax0, ax1), ((H0, W0), (H1, W1))):
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02)

    for p0, p1 in zip(kpts0, kpts1):
        con = ConnectionPatch(
            xyA=(p0[0], p0[1]), coordsA=ax0.transData,
            xyB=(p1[0], p1[1]), coordsB=ax1.transData,
            axesA=ax0, axesB=ax1, color=MPL_GREEN, linewidth=float(line_width), zorder=3
        )
        con.set_in_layout(False)
        fig.add_artist(con)
        if dot_size > 0:
            ax0.scatter([p0[0]], [p0[1]], s=dot_size**2, c=[MPL_GREEN], marker='o', zorder=4)
            ax1.scatter([p1[0]], [p1[1]], s=dot_size**2, c=[MPL_GREEN], marker='o', zorder=4)

    if outpath:
        fig.savefig(outpath, bbox_inches=None, pad_inches=0)
        plt.close(fig); return
    if show:
        plt.show()
    else:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        plt.close(fig)
        return buf
