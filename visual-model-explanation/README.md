![](https://img.shields.io/badge/library-fastai%20v1.0.42-ff69b4.svg)

Visual explanation of the model's decision making. As example: Fine-tuned ResNet34 as wild cats recognizer.

The web app can be deployed with
```
cd app
docker build -t wild-cats .
docker run -p 8000:8000 wild-cats
open http://localhost:8000
```

![](app.png)
