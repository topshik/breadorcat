# Breadorcat

## Model

Download model [here](https://drive.google.com/open?id=1fPazxMEYPXUc1YBMY_SGCYD_izGG23Ck)

`prepare_data.py` collects data from the folder `./source_data`, which consists of 4 folders containing unarchived open source datasets: [Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip), [Dogs vs. Cats](https://www.kaggle.com/c/3362/download-all), [Food-11](https://mmspg.epfl.ch/downloads/food-image-datasets/) and [iFood-2019](https://www.kaggle.com/c/13663/download-all).
**Note:** hierarchy `./data/{train, test}/{bread, cat, other}` must be preapred in advance.

Data collected with the script above can be downloaded [here](https://drive.google.com/open?id=1ySVtkBINUm6y1kHnUeYwjhTqo_4-bKVH)

## Backend

To start backend:

```bash
virtualenv .env --python=python3.7
. .env/bin/activate
pip install -r requirements.txt
python backend/main.py -p 8080
```

or

```bash
virtualenv .env --python=python3.7
. .env/bin/activate
pip install -r requirements.txt
make runserver
```
