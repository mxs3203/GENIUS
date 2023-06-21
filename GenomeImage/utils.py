import numpy as np
import pandas as pd

from GenomeImage.Image import Image
from GenomeImage.ImageCell import ImageCell

genes_per_chr = {'1': 3759, '10': 1593, '11': 2131, '12': 1867, '13': 930, '14': 1275,
                 '15': 1372, '16': 1530, '17': 1953, '18': 699, '19': 2057, '2': 2696,
                 '20': 1039, '21': 584, '22': 873, '23': 1348, '24': 162, '3': 2132,
                 '4': 1682, '5': 1851, '6': 2057, '7': 1896, '8': 1554, '9': 1622}


def normalize_data(data, min, max):
    return (data - min) / (max - min)


def make_image(id, met, all_genes, img_size=193):
    cnt = 0
    dict = {}
    for i in range(img_size+1):
        for j in range(img_size+1):
            if cnt < all_genes.shape[0]:
                img = ImageCell(all_genes['name2'].iloc[cnt], loss_val=None, gain_val=None, mut_val=None, exp_val=None,
                                methy_val=None, chr=all_genes['chr'].iloc[cnt])
                img.i = i
                img.j = j
                dict[all_genes['name2'].iloc[cnt]] = img
            else:
                img = ImageCell(None, None, None, None, None, None, None)
                img.i = i
                img.j = j
            cnt += 1
        cnt += 1
    return Image(id=id, met=met, dict_of_cells=dict)


def make_image_chr(id, met, all_genes):
    cnt = 0
    dict = {}
    for i, row in all_genes.iterrows():
        img = ImageCell(row['name2'], loss_val=None, gain_val=None,
                        mut_val=None, exp_val=None, methy_val=None,
                        chr=row['chr'])
        img.i = row['chr'] - 1
        img.j = cnt
        dict[row['name2']] = img
        limit = genes_per_chr[str(row['chr'])]
        if cnt >= limit:
            cnt = 0
        else:
            cnt += 1
    return Image(id=id, met=met, dict_of_cells=dict)


def find_mutations(id, image, muts):
    if id in np.array(muts['sampleID']):
        muts_tmp = muts.loc[muts['sampleID'] == id]
        for i, row in muts_tmp.iterrows():
            if row['Hugo_Symbol'] in image.dict_of_cells:
                image.dict_of_cells[row['Hugo_Symbol']].mut_val = row['PolyPhen_num']
        #print("\tFound ", muts_tmp.shape[0], "genes affected by mutation")
    return image


def find_gene_expression(id, image, gene_exp, min, max):
    if id in np.array(gene_exp.columns):
        genes = gene_exp['gene']
        exp = gene_exp[id]
        for i in range(len(genes)):
            if genes[i] in image.dict_of_cells:
               # print(exp[i], " -> ", normalize_data(exp[i], min,max))
                image.dict_of_cells[genes[i]].exp_val = normalize_data(exp[i], min, max)
    return image


def find_losses(id, image, all_genes, ascat_loss):
    if id in np.array(ascat_loss['ID']):
        ascat_loss_tmp = ascat_loss.loc[ascat_loss['ID'] == id]
        for i, row in ascat_loss_tmp.iterrows():
            seg_end = row['End']
            seg_start = row['Start']
            # find all affected genes
            genes_affected_full = all_genes['name2'][
                ((all_genes['start'] >= seg_start) & (all_genes['end'] <= seg_end))]
            genes_affected_partial1 = all_genes['name2'][all_genes['start'].between(seg_start, seg_end, inclusive='both')]
            genes_affected_partial2 = all_genes['name2'][all_genes['end'].between(seg_start, seg_end, inclusive='both')]

            genes_affected = pd.concat([genes_affected_full, genes_affected_partial1, genes_affected_partial2])
            # print("\tFound ", len(genes_affected_full), "genes affected by full loss")
            # print("\tFound ", len(genes_affected_partial1), "genes affected by partial loss(start)")
            # print("\tFound ", len(genes_affected_partial2), "genes affected by partial loss(end)")
            for g in genes_affected:
                if g in image.dict_of_cells:
                    #print(row['log_r'], " -> ", normalize_data(row['log_r'],  ascat_loss['log_r'].max(), ascat_loss['log_r'].min()))
                    image.dict_of_cells[g].loss_val = normalize_data(row['log_r'], ascat_loss['log_r'].max(),
                                                                     ascat_loss['log_r'].min())

    return image


def find_gains(id, image, all_genes, ascat_gains):
    if id in np.array(ascat_gains['ID']):
        ascat_loss_tmp = ascat_gains.loc[ascat_gains['ID'] == id]
        for i, row in ascat_loss_tmp.iterrows():
            seg_end = row['End']
            seg_start = row['Start']
            # find all affected genes
            genes_affected = all_genes['name2'][((all_genes['start'] >= seg_start) & (all_genes['end'] <= seg_end))]
            # print("\tFound ", len(genes_affected), "genes affected by gain")
            for g in genes_affected:
                if g in image.dict_of_cells:
                    #print(row['log_r'], " -> ", normalize_data(row['log_r'], ascat_gains['log_r'].min(), ascat_gains['log_r'].max()))
                    image.dict_of_cells[g].gain_val = normalize_data(row['log_r'], ascat_gains['log_r'].min(),
                                                                     ascat_gains['log_r'].max())

    return image


def find_methylation(id, image, methy):
    genes = methy['gene1']
    if id in np.array(methy.columns):
        meth = methy[id]
        for i in range(len(genes)):
            if genes[i] in image.dict_of_cells:
                image.dict_of_cells[genes[i]].methy_val = meth[i]
    return image


def makeImages(x):
    img_cin_g = x[0, :, :]
    img_cin_g = img_cin_g.astype('uint8')
    img_cin_l = x[ 1, :, :]
    img_cin_l = img_cin_l.astype('uint8')
    img_mut = x[2, :, :]
    img_mut = img_mut.astype('uint8')
    img_exp = x[3, :, :]
    img_exp = img_exp.astype('uint8')
    img_mety = x[4, :, :]
    img_mety = img_mety.astype('uint8')


    return img_cin_l, img_cin_g, img_mut, img_exp, img_mety
