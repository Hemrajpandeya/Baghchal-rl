# slow_gif.py
import sys
import numpy as np
import imageio.v2 as imageio
from PIL import Image

if len(sys.argv) < 4:
    print("Usage: python slow_gif.py <input.gif> <output.gif> <delay_seconds>")
    sys.exit(1)

inp, outp, delay = sys.argv[1], sys.argv[2], float(sys.argv[3])

# 1) Read all frames via reader (more robust than mimread for odd GIFs)
reader = imageio.get_reader(inp)
pil_frames = []
max_w = max_h = 0

for frame in reader:
    img = Image.fromarray(frame)
    if img.mode != "RGBA":
        img = img.convert("RGBA")  # force 4 channels
    pil_frames.append(img)
    w, h = img.size
    if w > max_w: max_w = w
    if h > max_h: max_h = h
reader.close()

# 2) Center-pad every frame to (max_w, max_h)
fixed = []
BG = (255, 255, 255, 255)  # white background
for img in pil_frames:
    if img.size != (max_w, max_h):
        canvas = Image.new("RGBA", (max_w, max_h), BG)
        x = (max_w - img.size[0]) // 2
        y = (max_h - img.size[1]) // 2
        canvas.paste(img, (x, y))
        img = canvas
    fixed.append(np.array(img))  # back to ndarray

# 3) Save slower GIF (duration = seconds per frame)
imageio.mimsave(outp, fixed, duration=delay, loop=0)
print(f"Saved {outp} with delay={delay}s/frame (~{1/delay:.2f} FPS) and size={max_w}x{max_h}.")
