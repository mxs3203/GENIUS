import argparse
import os
import sys

import pandas as pd
import numpy as np
import pickle
import time

from tqdm import tqdm

sys.path.insert(0, sys.path[0] + "/..")

from GenomeImage.utils import make_image, find_losses, find_gains, find_mutations, find_gene_expression, \
    find_methylation

if not os.path.exists("../data/example_data/genome_images"):
    # If it doesn't exist, create it
    os.makedirs("../data/example_data/genome_images")

DEBUG = False

parser = argparse.ArgumentParser(description='')
parser.add_argument('--clinical_data',type=str, default='../data/example_data/clinical.csv')
parser.add_argument('--ascat_data',type=str, default='../data/example_data/ascat.csv')
parser.add_argument('--all_genes_included',type=str, default='../data/example_data/all_genes_ordered_by_chr_no_sex_chr.csv')
parser.add_argument('--mutation_data',type=str, default='../data/example_data/muts.csv')
parser.add_argument('--gene_exp_data',type=str, default='../data/example_data/gene_exp_matrix.csv')
parser.add_argument('--gene_methyl_data',type=str, default='../data/example_data/methylation.csv')
args = parser.parse_args()
print(args)
start_time = time.time()
print("Reading clinical...")
clinical = pd.read_csv(args.clinical_data)
print("Reading ascat...")
ascat = pd.read_csv(args.ascat_data)
ascat_loss = ascat.loc[ascat['loss'] == True]
ascat_gain = ascat.loc[ascat['gain'] == True]
print("Reading all gene definition...")
all_genes = pd.read_csv(args.all_genes_included)

print("Reading Muts...")
muts = pd.read_csv(args.mutation_data)
print("Reading gene exp...")
gene_exp = pd.read_csv(args.gene_exp_data)
print("Reading Methylation...")
methy = pd.read_csv(args.gene_methyl_data)

for index, row in tqdm(clinical.iterrows(), total=len(clinical)):

    id = row['bcr_patient_barcode']
    if DEBUG:
        print(id)
        print(index, "/", clinical.shape[0], "  ", id)
    met = row['metastatic_one_two_three']

    tmp_mut = muts[muts["sampleID"] == id]

    image = make_image(id, met, all_genes)
    image = find_losses(id, image, all_genes, ascat_loss)
    image = find_gains(id, image, all_genes, ascat_gain)
    image = find_mutations(id, image, tmp_mut)
    image = find_gene_expression(id, image, gene_exp,
                                 np.min(np.array(gene_exp.select_dtypes(include=np.number))),
                                 np.max(np.array(gene_exp.select_dtypes(include=np.number))))
    image = find_methylation(id, image, methy)
    image.make_image_matrces()
    five_dim_image = image.make_5_dim_image()
    feature_vector = image.vector_of_all_features()
    if np.all((feature_vector == 0)):
        print("All zeros in 5d, not saving...")
    else:
        with open("../data/example_data/genome_images/{}.dat".format(id),
                  'wb') as f:
            pickle.dump(five_dim_image, f)
            f.close()




print("Done in --- %s minutes ---" % ((time.time() - start_time) / 60))
