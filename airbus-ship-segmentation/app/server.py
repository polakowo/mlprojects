from starlette.applications import Starlette
from starlette.templating import Jinja2Templates
import uvicorn
import aiohttp
import asyncio
import sys
from io import BytesIO
from PIL import Image
import base64
import cv2
import numpy as np
import torch
from fastai import vision
from skimage.morphology import label

# Required to unpickle the model
from unpickler_attrs import *

path = vision.Path(__file__).parent
app = Starlette(debug=True)
templates = Jinja2Templates(str(path/'templates'))


def load_learner(path, fname):
    # Load the model in cpu mode
    # doesn't work: defaults.device = torch.device('cpu')
    # doesn't work: torch.cuda.set_device('cpu')
    state = torch.load(vision.Path(path)/fname, map_location='cpu')
    model = state.pop('model')
    src = vision.LabelLists.load_state(path, state.pop('data'))
    data = src.databunch()
    cb_state = state.pop('cb_state')
    clas_func = state.pop('cls')
    res = clas_func(data, model, **state)
    res.callback_fns = state['callback_fns']  # to avoid duplicates
    res.callbacks = [vision.load_callback(c, s, res) for c, s in cb_state.items()]
    return res


async def setup_learners():
    # Export your learner with learn.export() and copy to the app/models folder
    resnet_learn = load_learner(path/'models', fname='Resnet34_256.pkl')
    unet_learn = load_learner(path/'models', fname='Unet34_256.pkl')
    return resnet_learn, unet_learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learners())]
resnet_learn, unet_learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def mask_overlay(image, mask, color=(0, 1, 0)):
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(image.dtype)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, np.argmax(color)] > 0
    img[ind] = weighted_sum[ind]
    return img


def image_from_tensor(img_tensor):
    numpied = img_tensor.squeeze()
    numpied = np.moveaxis(numpied.detach().numpy(), 0, -1)
    numpied = numpied - np.min(numpied)
    numpied = numpied/np.max(numpied)
    return numpied


def rle_encode(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def multi_rle_encode(img):
    labels = label(img)
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


def count_ships(mask_tensor):
    mask_tensor = mask_tensor.argmax(dim=1)
    mask_tensor = mask_tensor.squeeze(0)
    masks = multi_rle_encode(mask_tensor)
    return len(masks)


async def get_prediction(file):
    # Bytes to image
    bytes = await (file.read())
    img = vision.open_image(BytesIO(bytes))
    # Does the image have ships?
    with_ships, _, _ = resnet_learn.predict(img)
    with_ships = str(with_ships) in ('True', '1')
    if with_ships:
        # Pre-process image (align shape etc.)
        img_tensor = unet_learn.data.one_item(img)[0]
        # Predict mask
        mask_tensor = unet_learn.model(img_tensor)
        mask_tensor = torch.softmax(mask_tensor, dim=1)
        ships = count_ships(mask_tensor)
        mask_tensor = mask_tensor[:, 1, ...]
        mask_tensor = mask_tensor.permute(0, 2, 1)
        # Post-process both tensors
        img = image_from_tensor(img_tensor)
        mask_img = image_from_tensor(mask_tensor)
        # Create overlay image
        overlay_img = mask_overlay(img, mask_img)
        overlay_img *= 255
        overlay_img = overlay_img.astype(np.uint8)
        overlay_img = Image.fromarray(overlay_img)
        # Decode overlay image into a base64 string
        buffered = BytesIO()
        overlay_img.save(buffered, format='PNG')
        image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {
            'image': image,
            'fname': file.filename,
            'ships': ships
        }
    else:
        image = base64.b64encode(bytes).decode("utf-8")
        return {
            'image': image,
            'fname': file.filename,
            'ships': 0
        }


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    predictions = list()
    for file in data.getlist('files'):
        predictions.append(await get_prediction(file))
    return templates.TemplateResponse("predict.html", {"items": predictions, "request": request})


@app.route("/")
def form(request):
    return templates.TemplateResponse("upload.html", {"request": request})


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8000)
