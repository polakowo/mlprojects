from starlette.applications import Starlette
from starlette.templating import Jinja2Templates
import uvicorn
import aiohttp
import asyncio
import sys

from io import BytesIO
from PIL import Image
from fastai import vision, callbacks
from scipy import ndimage
import cv2
import base64
import numpy as np
import matplotlib.pyplot as plt

path = vision.Path(__file__).parent
app = Starlette(debug=True)
templates = Jinja2Templates(str(path/'templates'))


async def setup_learner():
    # Export your learner with learn.export() and copy to the app/models folder
    return vision.load_learner(path/'models', fname='Resnet34.pkl')

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


def hooked_backward(xb):
    m = learn.model.eval()
    with callbacks.hook_output(m[0]) as hook_a:
        with callbacks.hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)[0]
            pred = vision.torch.argmax(preds)
            preds[pred].backward()
    return hook_a, hook_g, preds


def upsample(im, to_shape):
    heatmap = ndimage.zoom(im, (to_shape[0] / im.shape[0], to_shape[1] / im.shape[1]), order=1)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    return heatmap


async def get_prediction(file):
    bytes = await (file.read())
    img = vision.open_image(BytesIO(bytes))

    # Preprocess the image for prediction
    xb, _ = learn.data.one_item(img)
    xb_im = np.moveaxis(learn.data.denorm(xb)[0].numpy(), 0, -1)

    # Retrieve the feature maps and the gradients
    hook_a, hook_g, preds = hooked_backward(xb)
    probs = vision.torch.softmax(preds, dim=0)
    label, prob = sorted(zip(learn.data.classes, map(float, probs)), key=lambda p: p[1], reverse=True)[0]
    acts = hook_a.stored[0].cpu()
    grad = hook_g.stored[0][0].cpu()

    # Process the heatmap
    grad_chan = grad.mean(1).mean(1)
    heatmap = (acts * grad_chan[..., None, None]).mean(0).numpy()
    heatmap = upsample(heatmap, xb_im.shape[:2])
    heatmap = plt.cm.plasma(heatmap)[..., :3]

    # Overlay the image and the heatmap
    heatmap = heatmap.astype(xb_im.dtype)
    overlay_img = cv2.addWeighted(xb_im, 0.5, heatmap, 0.5, 0.)
    overlay_img *= 255
    overlay_img = overlay_img.astype(np.uint8)
    overlay_img = Image.fromarray(overlay_img)

    # Prepare the output
    buffered = BytesIO()
    overlay_img.save(buffered, format='PNG')
    image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {
        'image': image,
        'fname': file.filename,
        'label': label,
        'prob': "{0:.2f}".format(prob)
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
