import glob
import pickle
import random
from collections import Counter

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler



class TCGAImageLoader(Dataset):

    def __init__(self, csv_file,  folder, image_type, predictor_column, response_column, filter_by_type=None, transform=None ):

        self.annotation = pd.read_csv(csv_file, sep=",")
        if filter_by_type is not None:
            self.annotation = self.annotation[self.annotation.type.isin(filter_by_type)]
        self.number_of_c_types = len(self.annotation['type'].unique())
        ord_enc = OrdinalEncoder()
        scaler = MinMaxScaler()
        self.annotation["type_coded"] = ord_enc.fit_transform(self.annotation[["type"]])
        self.annotation["type_coded_random"] = np.random.randint(0,self.number_of_c_types, size=np.shape(self.annotation)[0])
        self.f_names = pd.unique(self.annotation['type'])
        self.transform = transform
        self.folder = folder
        self.image_type = image_type
        self.predictor_column = predictor_column
        self.response_column = response_column
        self.remove_rows_where_there_is_no_file()

    def compute_class_weight(self, dataset):
        y = []
        for x,y_label,i in dataset:
            y.append(y_label.item())
        count = Counter(y)
        class_count = np.array([count[0], count[1]])
        weight = 1. / class_count
        samples_weight = np.array([weight[t] for t in y])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def remove_rows_where_there_is_no_file(self):
        print("Finding all files from metadata... number of files: ",np.shape(self.annotation)[0])
        files = glob.glob("{}/*.dat".format(self.folder))
        ids = [f.split("/")[4] for f in files]
        ids = [f.split(".")[0] for f in ids]
        self.annotation = self.annotation[self.annotation['bcr_patient_barcode'].isin(ids)]
        print("Number of Files after removing the missing files: ",np.shape(self.annotation)[0])

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        with open("{}/{}.dat".format(self.folder, self.annotation.iloc[idx, self.predictor_column]), 'rb') as f:
            x = pickle.load(f)
            f.close()
        y = np.array(self.annotation.iloc[idx, self.response_column], dtype="long")
        if self.transform:
            x = self.transform(x)

        return x, y, self.annotation.iloc[idx, self.predictor_column]