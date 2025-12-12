import os

import numpy as np
from fastapi import FastAPI, Response, File, UploadFile
from scipy.ndimage import gaussian_filter
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
from PIL import Image
import io

from training.train import img_size

app = FastAPI()
model_path = 'model.keras'
colormap = {
    0: (0, 0, 0),
    1: (45, 22, 64),
    2: (229, 177, 147),
    3: (66, 36, 7),
    4: (66, 36, 7),
    5: (255, 255, 255),
    6: (255, 255, 255),
    7: (227, 127, 133),
    8: (255, 255, 255),
}

if os.path.exists(model_path):
    try:
        model = load_model(model_path)
        print('Model loaded!')
    except Exception as e:
        print(f'Error: {e}')
        model = None
else:
    print('Model not found')
    model = None


def get_mask(image):
    orig_width, orig_height = image.size

    image = image.resize((img_size, img_size))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr)
    mask = tf.argmax(prediction, axis=-1)
    mask = mask[..., tf.newaxis]
    mask = mask[0]
    mask = tf.image.resize(mask, (orig_height, orig_width), method='nearest')

    return mask


@app.post('/api/mask')
async def mask(file: UploadFile = File(...)):
    image_raw = await file.read()
    image = Image.open(io.BytesIO(image_raw)).convert('RGB')

    mask = np.array(get_mask(image))
    mask = np.squeeze(mask)
    color_mask = np.zeros((image.height, image.width, 3), dtype=np.uint8)

    for cls, color in colormap.items():
        color_mask[mask == cls] = color
    mask = Image.fromarray(color_mask)

    out_bytes = io.BytesIO()
    mask.save(out_bytes, format='PNG')
    out_bytes.seek(0)

    return Response(content=out_bytes.read(), media_type='image/png')


@app.post('/api/background')
async def background(file: UploadFile = File(...)):
    image_raw = await file.read()
    image = Image.open(io.BytesIO(image_raw)).convert('RGB')

    mask = np.array(get_mask(image))
    mask = np.squeeze(mask)
    binary_mask = (mask > 0).astype(np.uint8)
    blurred_mask = gaussian_filter(binary_mask.astype(float), sigma=3)

    image_array = np.array(image.convert('RGBA'))
    image_array[..., 3] = (blurred_mask * 255).astype(np.uint8)
    new_image = Image.fromarray(image_array, 'RGBA')

    out_bytes = io.BytesIO()
    new_image.save(out_bytes, format='PNG')
    out_bytes.seek(0)

    return Response(content=out_bytes.read(), media_type='image/png')


@app.post('/api/teeth')
async def eyes(file: UploadFile = File(...)):
    image_raw = await file.read()
    image = Image.open(io.BytesIO(image_raw)).convert('RGB')
    mask = np.array(get_mask(image)).squeeze()

    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    coarse_mask = np.zeros((h, w), dtype=np.uint8)
    coarse_mask[mask == 8] = 255

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    _, l_mask = cv2.threshold(l_channel, 60, 255, cv2.THRESH_BINARY)
    _, not_red_mask = cv2.threshold(a_channel, 145, 255, cv2.THRESH_BINARY_INV)

    refined_mask = cv2.bitwise_and(coarse_mask, l_mask)
    refined_mask = cv2.bitwise_and(refined_mask, not_red_mask)

    refined_mask_blurred = cv2.GaussianBlur(refined_mask, (15, 15), 0)
    alpha = refined_mask_blurred.astype(float) / 255.0

    l_float = l_channel.astype(float)
    b_float = b_channel.astype(float)

    brightness_factor = 0.5
    l_whitened = np.clip(l_float + brightness_factor, 0, 255)

    yellow_reduction_factor = 0.3
    b_whitened = (b_float - 128.0) * yellow_reduction_factor + 128.0
    b_whitened = np.clip(b_whitened, 0, 255)

    lab_whitened = cv2.merge([
        l_whitened.astype(np.uint8),
        a_channel,
        b_whitened.astype(np.uint8)
    ])
    img_bgr_whitened = cv2.cvtColor(lab_whitened, cv2.COLOR_LAB2BGR)

    final_bgr = np.empty_like(img_bgr)
    for i in range(3):
        final_bgr[:, :, i] = (img_bgr[:, :, i] * (1 - alpha) +
                              img_bgr_whitened[:, :, i] * alpha).astype(np.uint8)

    result_image = Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))

    out_bytes = io.BytesIO()
    result_image.save(out_bytes, format='PNG')
    out_bytes.seek(0)
    return Response(content=out_bytes.read(), media_type='image/png')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
