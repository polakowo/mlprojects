![](https://img.shields.io/badge/library-fastai%20v1.0.42-ff69b4.svg)

Dynamic Unet with ResNet34 backbone for the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection) competition.

The web app can be deployed with
```
cd app
docker build -t ship-detection .
docker run -p 8000:8000 ship-detection
open http://localhost:8000
```

![](app.png)
