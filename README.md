# Music emotion recognition
Using Deep Learning for emotion analysis in songs.

Inspired by the work of Jan Jakubik, Halina Kwasnicka, *Music Emotion Analysis Using Semantic Embedding Recurrent Neural Networks*.

## Dataset

Download datasets and add them in the `data` folder, matching paths defined in `config.py`.

Emotify dataset can be found [here](https://drive.google.com/open?id=1Jq4zpt0tMQyAe8apuqIs5J097ZW8XjcJ).
Download it and store it under `data/emotifymusic`.

## Trainning models

- Choose the model in `config.py`
- Run `python processing.py` to process the data

To train with validation:
- Run `python split.py --validate` to generate train/validate datasets
- Run `python train.py {model_name} --validate` to train the selected model on the train dataset and validate it

To train and test:
- Run `python split.py` to generate train and test datasets
- Run `python train.py {model_name}` to train the selected model on the train dataset without validating it
- Run `python test.py {path_to_model}` to test the model saved at `path_to_model` on the test dataset

## Requirements

Make sure you have `ffmpeg` installed locally.
- `sudo apt install ffmpeg` on Ubuntu.
- `brew install ffmpeg` on macOS.

`pip install -r requirements.txt`

To update requirements.txt:
`pip freeze > requirements.txt`