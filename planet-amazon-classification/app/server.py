from starlette.applications import Starlette
from starlette.templating import Jinja2Templates
import uvicorn
import aiohttp
import asyncio
import sys
from io import BytesIO
from PIL import Image
from fastai import vision
import base64

path = vision.Path(__file__).parent
app = Starlette(debug=True)
templates = Jinja2Templates(str(path/'templates'))


async def setup_learner():
    # Export your learner with learn.export() and copy to the app/models folder
    return vision.load_learner(path/'models', fname='Resnet50_256.pkl')

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


async def get_prediction(file):
    bytes = await (file.read())
    img = vision.open_image(BytesIO(bytes))
    labels, _, _ = learn.predict(img)
    image = base64.b64encode(bytes).decode("utf-8")
    return {
        'image': image,
        'fname': file.filename,
        'labels': labels.obj
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
