from starlette.applications import Starlette
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
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

# PosixPath
path = vision.Path(__file__).parent

resnet_file_name = 'Resnet34_256.pkl'
unet_file_name = 'Unet34_256.pkl'

resnet_file_url = f'https://www.dropbox.com/s/a7t8i40hzna1200/{resnet_file_name}?raw=1'
unet_file_url = f'https://www.dropbox.com/s/r83mgdq2y7dfhxt/{unet_file_name}?raw=1'

# Starlette config
app = Starlette(debug=True)
app.mount('/static', StaticFiles(directory='app/static'))
templates = Jinja2Templates(str(path/'templates'))

async def download_file(url, dest):
    """Download a file."""
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


def load_learner(path, fname):
    """Load the inference learner from disk."""
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
    """Download and load all inference learners."""
    await download_file(resnet_file_url, path/"models"/resnet_file_name)
    await download_file(unet_file_url, path/"models"/unet_file_name)
    # Export your learner with learn.export() and copy to the app/models folder
    resnet_learn = load_learner(path/'models', fname='Resnet34_256.pkl')
    unet_learn = load_learner(path/'models', fname='Unet34_256.pkl')
    return resnet_learn, unet_learn

# Set up everything once
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learners())]
resnet_learn, unet_learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


async def get_bytes(url):
    """Get bytes from URL."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def mask_overlay(image, mask, color=(1, 0.5, 0)):
    """Overlay image array with mask array."""
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(image.dtype)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, np.argmax(color)] > 0
    img[ind] = weighted_sum[ind]
    return img


def image_from_tensor(img_tensor):
    """Post-process the image tensor for the use in PIL."""
    numpied = img_tensor.squeeze()
    numpied = np.moveaxis(numpied.detach().numpy(), 0, -1)
    numpied = numpied - np.min(numpied)
    numpied = numpied/np.max(numpied)
    return numpied


def rle_encode(img):
    """Run-length encode a single ship."""
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def multi_rle_encode(img):
    """Run-length encode all ships in the image."""
    labels = label(img)
    return [rle_encode(labels == k) for k in np.unique(labels[labels > 0])]


def count_ships(mask_tensor):
    """Count the number of ships in the mask tensor."""
    mask_tensor = mask_tensor.argmax(dim=1)
    mask_tensor = mask_tensor.squeeze(0)
    masks = multi_rle_encode(mask_tensor)
    return len(masks)


async def get_prediction(vision_img, filename):
    """Generate the prediction and construct the JSON response."""
    # Does the image have ships? -> Let's let ResNet decide first, saves some time
    with_ships, _, _ = resnet_learn.predict(vision_img)
    with_ships = str(with_ships) in ('True', '1')
    if with_ships:
        # Pre-process image (align shape etc.)
        img_tensor = unet_learn.data.one_item(vision_img)[0]
        # Predict the mask
        mask_tensor = unet_learn.model(img_tensor)
        mask_tensor = torch.softmax(mask_tensor, dim=1)
        ships = count_ships(mask_tensor)
        mask_tensor = mask_tensor[:, 1, ...]
        mask_tensor = mask_tensor.permute(0, 2, 1)
        # Post-process both tensors
        img = image_from_tensor(img_tensor)
        mask_img = image_from_tensor(mask_tensor)
        # Overlay the mask on the original image
        overlay_img = mask_overlay(img, mask_img)
        overlay_img *= 255
        overlay_img = overlay_img.astype(np.uint8)
        overlay_img = Image.fromarray(overlay_img)
        # Decode the resulting image into a base64 string
        buffered = BytesIO()
        overlay_img.save(buffered, format='PNG')
        image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {
            'image': image,
            'fname': filename,
            'ships': ships
        }
    else:
        buffered = BytesIO()
        img_np = vision.image2np(vision_img.data*255).astype(np.uint8)
        img = Image.fromarray(img_np).resize((256, 256))
        img.save(buffered, format='PNG')
        image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return {
            'image': image,
            'fname': filename,
            'ships': 0
        }


@app.route("/upload", methods=["POST"])
async def upload(request):
    """User hits the upload button."""
    data = await request.form()
    file = data['file']
    # Bytes to image
    bytes = await (file.read())
    vision_img = vision.open_image(BytesIO(bytes))
    prediction = await get_prediction(vision_img, file.filename)
    return templates.TemplateResponse("predict.html", {"item": prediction, "request": request})


@app.route("/select", methods=["GET"])
async def example(request):
    """User selects an example image."""
    params = dict(request.query_params)
    vision_img = vision.open_image(path/"static"/"img"/params['fname'])
    prediction = await get_prediction(vision_img, params['fname'])
    return templates.TemplateResponse("predict.html", {"item": prediction, "request": request})


@app.route("/")
def form(request):
    """User calls the index page."""
    return templates.TemplateResponse("upload.html", {"request": request})


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
