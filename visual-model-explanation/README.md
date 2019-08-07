### Visual explanation of the model's decision making

The Jupyter Notebook scrapes the dataset from Google Images, detects duplicates using embeddings, provides visual explanation of the model using [Grad-CAM](https://arxiv.org/abs/1610.02391) and [lime](https://github.com/marcotcr/lime), and clusters embeddings using t-SNE and other techniques.

The web app can be deployed with
```
cd app
docker build -t wild-cats .
docker run -p 8000:8000 wild-cats
open http://localhost:8000
```

![Web app screenshot](app.png)
