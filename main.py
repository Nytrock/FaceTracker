import os
from typing import List

import numpy as np
from fastapi import FastAPI, Response, File, UploadFile, Form
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import io

from training.train import img_size

app = FastAPI()
model_path = 'model.keras'

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


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


@app.post('/api/prediction')
async def predict(
        image: UploadFile = File(...),
        checks: List[str] = Form(None)
):
    # Чтение изображение
    original_bytes = await image.read()
    original_img = Image.open(io.BytesIO(original_bytes)).convert('RGB')
    orig_width, orig_height = original_img.size

    img_128 = original_img.resize((img_size, img_size))
    arr = np.array(img_128, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prediction = model.predict(arr)
    mask_128 = create_mask(prediction)

    mask_img_128 = tf.keras.utils.array_to_img(mask_128)
    mask_resized = mask_img_128.resize((orig_width, orig_height), Image.BILINEAR)

    out_bytes = io.BytesIO()
    mask_resized.save(out_bytes, format='PNG')
    out_bytes.seek(0)

    return Response(content=out_bytes.read(), media_type='image/png')


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)
