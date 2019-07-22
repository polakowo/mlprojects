Similar to the Airbus competition, the [Planet: Understanding the Amazon from Space challenge](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) was about processing satellite images. Here, the image chips with various atmospheric conditions and classes of land cover/land use had to be tagged (i.e. a multi-label classification task). We trained a simple, fine-tuned ResNet50 to achieve good performance.

The web app can be deployed with
```
cd app
docker build -t planet-amazon .
docker run -p 8000:8000 planet-amazon
open http://localhost:8000
```

![](app.png)
