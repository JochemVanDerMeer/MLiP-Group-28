# Loads the trained arcface model and generates a list of vector embeddings for each image in the training set

import os
from test import get_featurs, get_lfw_list
from config.config import Config
from torch.nn import DataParallel
from models.resnet import *

inputfilepath = "/scratch/guchoadeassis/groupw/preprocessed_mlip_train_data"
outputfilepath = "/scratch/guchoadeassis/groupw/outputvectors.txt"
dirs = os.listdir(inputfilepath)

image_list, label_list = [], []

for file in dirs:
    filepath = os.path.join(inputfilepath, file)
    if not file.startswith('.'):
        for image in os.listdir(filepath):
            imagepath = os.path.join(filepath, image)
            image_list.append(imagepath)
            label_list.append(file)

opt = Config()
model = resnet_face18(opt.use_se)

model = DataParallel(model)
model.load_state_dict(torch.load(opt.test_model_path))
model.to(torch.device("cuda"))

identity_list = get_lfw_list(opt.lfw_test_list)
img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

model.eval()
with open(outputfilepath, "w") as f:
    fts, _ = get_featurs(model, image_list, 10)
    for i, ft in enumerate(fts):
        line = f'{label_list[i]}: {list(ft)}\n'
        f.write(line)