import argparse
import importlib
import json
import sys
from tqdm import tqdm
import captum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, sys.path[0] + "/..")

from GenomeImage.utils import make_image
from Training.Dataloader import TCGAImageLoader
from Training.Models.AE_Square import AE

parser = argparse.ArgumentParser(description='')
parser.add_argument('--genome_images',type=str, default="../data/tcga/genome_images")
parser.add_argument('--config_file',type=str, default="../Training/config")
parser.add_argument('--clinical_data',type=str, default="../data/tcga/clinical.csv")
parser.add_argument('--model_checkpoint',type=str, default="../Training/saved_models/best_model.pb")
parser.add_argument('--all_genes_file',type=str, default="../data/tcga/all_genes_ordered_by_chr_no_sex_chr.csv")
parser.add_argument('--attribution_n_steps',type=str, default=10)
args = parser.parse_args()

with open(args.config_file, "r") as jsonfile:
    config = json.load(jsonfile)
    print("Read successful")

image_type = config['image_type']
folder = args.genome_images
predictor_column = config['predictor_column']
response_column = config['response_column']
metadata = args.clinical_data
checkpoint_file = args.model_checkpoint
all_genes_file = args.all_genes_file


all_genes = pd.read_csv(all_genes_file)
# Model Params
net = AE(output_size=2)

def wrapped_model(inp):
    return net(inp)[0]

checkpoint = torch.load(checkpoint_file)
optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=config['lr_decay'], lr=config['LR'], weight_decay=config['weight_decay'])
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
net.eval()
# make IG Instance of a model
occlusion = captum.attr.IntegratedGradients(wrapped_model)
cancer_types = ['OV', 'BLCA','KIRC', 'STAD', 'UCEC']



# Run for every cancer type specified in a list
for type in cancer_types:

    type = str(type)
    # load the data
    dataset = TCGAImageLoader(metadata,
                              folder,
                              image_type,
                              predictor_column,
                              response_column,
                              filter_by_type=[type])
    if dataset.annotation.shape[0] != 0:
        trainLoader = DataLoader(dataset, batch_size=1, num_workers=10, shuffle=False)
        print(type, "Samples: ", len(trainLoader))
        # prepare empty lists
        heatmaps_meth = []
        heatmaps_loss = []
        heatmaps_gains = []
        heatmaps_mut = []
        heatmaps_exp = []

        # iterate sample by samples
        for x, y_dat , id in tqdm(trainLoader):
            if y_dat == 1:
                baseline = torch.zeros((1, x.shape[1], x.shape[2], x.shape[3]))
                attribution = occlusion.attribute(x, baseline, target=1, n_steps=int(args.attribution_n_steps))
                attribution = attribution.squeeze(0).cpu().detach().numpy()
                heatmaps_gains.append(np.abs(attribution[0, :, :]))
                heatmaps_loss.append(np.abs(attribution[1, :, :]))
                heatmaps_mut.append(np.abs(attribution[2, :, :]))
                heatmaps_exp.append(np.abs(attribution[3, :, :]))
                heatmaps_meth.append(np.abs(attribution[4, :, :]))

        heatmaps_loss = np.array(heatmaps_loss)
        mean_loss_matrix = heatmaps_loss.mean(axis=0)
        ax = sns.heatmap(mean_loss_matrix, cmap="YlGnBu")
        plt.title("{}_{}".format(type, "Losses"))
        plt.savefig("{}_{}.png".format(type, "Losses"), dpi=200)

        heatmaps_gains = np.array(heatmaps_gains)
        mean_gain_matrix = heatmaps_gains.mean(axis=0)
        ax = sns.heatmap(mean_gain_matrix, cmap="YlGnBu")
        plt.title("{}_{}".format(type, "Gains"))
        plt.savefig("{}_{}.png".format(type, "Gains"), dpi=200)

        heatmaps_mut = np.array(heatmaps_mut)
        mean_mut_matrix = heatmaps_mut.mean(axis=0)
        ax = sns.heatmap(mean_mut_matrix, cmap="YlGnBu")
        plt.title("{}_{}".format(type, "Muts"))
        plt.savefig("{}_{}.png".format(type, "Muts"), dpi=200)

        heatmaps_exp = np.array(heatmaps_exp)
        mean_exp_matrix = heatmaps_exp.mean(axis=0)
        ax = sns.heatmap(mean_exp_matrix, cmap="YlGnBu")
        plt.title("{}_{}".format(type, "Exp"))
        plt.savefig("{}_{}.png".format(type, "Expression"), dpi=200)

        heatmaps_meth = np.array(heatmaps_meth)
        mean_meth_matrix = heatmaps_meth.mean(axis=0)
        ax = sns.heatmap(mean_meth_matrix, cmap="YlGnBu")
        plt.title("{}_{}".format(type, "Methylation"))
        plt.savefig("{}_{}.png".format(type, "Methylation"), dpi=200)

        number_of_genes_returned = all_genes.shape[0]-1

        image = make_image("ID", 1, all_genes)
        exp_att = image.analyze_attribution(mean_exp_matrix, number_of_genes_returned, "Expression")
        mut_att = image.analyze_attribution(mean_mut_matrix, number_of_genes_returned, "Mutation")
        gain_att = image.analyze_attribution(mean_gain_matrix, number_of_genes_returned, "Gain")
        loss_att = image.analyze_attribution(mean_loss_matrix, number_of_genes_returned, "Loss")
        meth_att = image.analyze_attribution(mean_meth_matrix, number_of_genes_returned, "Methylation")

        total_df = pd.concat([exp_att,mut_att,gain_att,loss_att, meth_att])

        total_df.to_csv("{}.csv".format(type))
        print("Attribution scores are saved in {}.csv file. Each data source also produced a PNG file representing "
              "attribution of each genome image channel.".format(type))

