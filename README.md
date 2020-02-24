# Music emotion recognition
Using Deep Learning for emotion analysis in songs.

Inspired by the work of Jan Jakubik, Halina Kwasnicka, *Music Emotion Analysis Using Semantic Embedding Recurrent Neural Networks*.

## Dataset

Download datasets and add them in the `data` folder, matching paths defined in `config.py`.

## Trainning models

- Choose the model in `config.py`
- Run `python processing.py` to process the data
- Run `python split.py` to generate train/test split
- Run `python main.py` to train the selected model on the dataset

## Requirements

Make sure you have `ffmpeg` installed locally.
- `sudo apt install ffmpeg` on Ubuntu.
- `brew install ffmpeg` on macOS.

`pip install -r requirements.txt`

To update requirements.txt:
`pip freeze > requirements.txt`