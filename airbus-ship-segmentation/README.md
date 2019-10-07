<img width=100 src="https://upload.wikimedia.org/wikipedia/de/thumb/f/fe/Airbus_Logo.svg/2000px-Airbus_Logo.svg.png"/>

#### Overview

This project is part of the [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection) and aims to support maritime monitoring services by automatically extracting objects from satellite images. In particular, the objective is to use a SOTA deep learning model to locate all ships in satellite images and put an aligned bounding box segment around each detected ship. The result of this project is a deployed web application capable of locating ships on the images uploaded by the user.

Tags: *Competition, Satellite Imagery, Ship Detection, Image Segmentation, Neural Networks, Transfer Learning, Python, fastai, Web App*

#### Files

- [Capstone proposal](CapstoneProposal.pdf)
- [Notebook with data preparation, training and submission](https://nbviewer.jupyter.org/github/polakowo/mlprojects/blob/master/airbus-ship-segmentation/ShipDetection.ipynb)
- [Final report](FinalReport.pdf)
- [Web application directory](app)

#### App deployment

- Open and run the [ShipDetection.ipynb](https://nbviewer.jupyter.org/github/polakowo/mlprojects/blob/master/airbus-ship-segmentation/ShipDetection.ipynb) notebook to produce the model artifacts.
- Upload the `Resnet34_256.pkl` and `Unet34_256.pkl` to Dropbox and copy their share links.
- Paste those links to `app/server.py` and change `dl=0` to `raw=1` at the end of each link.
- To deploy the web application, enter the following commands in console:

```
cd app
docker build -t ship-detection .
docker run -p 5000:5000 ship-detection
open http://localhost:5000
```

![Web app screenshot](app.png)
