import os
import numpy as np
from PIL import Image

INPUT_DIR = "main/annotations_mine"

CLASS_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 4,
    6: 4,
    7: 5,
    8: 6
}

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(".png"):
        continue

    path = os.path.join(INPUT_DIR, filename)
    mask = Image.open(path)
    mask_np = np.array(mask)
    new_mask = np.copy(mask_np)

    for old_class, new_class in CLASS_MAP.items():
        new_mask[mask_np == old_class] = new_class
    Image.fromarray(new_mask.astype(np.uint8)).save(path)
print("Готово.")
