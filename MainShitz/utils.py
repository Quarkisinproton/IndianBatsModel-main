"""
utils.py - The junk drawer of helper functions

Model I/O and data loading utilities. Nothing fancy,
just the stuff that didn't fit anywhere else.
"""
import os
import torch
from PIL import Image


def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


def save_model(model, model_path):
    torch.save(model, model_path)


def load_data(data_path):
    #Load all images from a directory.]
    images = []
    labels = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                image = Image.open(img_path)
                images.append(image)
                labels.append(label)
    return images, labels