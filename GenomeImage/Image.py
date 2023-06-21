import itertools

import numpy as np
import pandas as pd


class Image:
    def __init__(self, id,dict_of_cells,met):
        self.id = id
        self.dict_of_cells = dict_of_cells
        self.met = met

        self.gain_matrix = None
        self.loss_matrix = None
        self.methy_matrix = None
        self.mut_matrix = None
        self.exp_matrix = None

        self.chr_gain_matrix = None
        self.chr_loss_matrix = None
        self.chr_methy_matrix = None
        self.chr_mut_matrix = None
        self.chr_exp_matrix = None

    def make_image_matrces(self):
        img_size = 194
        loss = np.zeros((img_size, img_size))
        gain = np.zeros((img_size, img_size))
        mut = np.zeros((img_size, img_size))
        methy = np.zeros((img_size, img_size))
        exp = np.zeros((img_size, img_size))

        for gene in self.dict_of_cells:
            if self.dict_of_cells[gene].loss_val is not None:
                loss[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].loss_val
            if self.dict_of_cells[gene].gain_val is not None:
                gain[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].gain_val
            if self.dict_of_cells[gene].mut_val is not None:
                mut[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].mut_val
            if self.dict_of_cells[gene].exp_val is not None:
                exp[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].exp_val
            if self.dict_of_cells[gene].methy_val is not None:
               methy[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].methy_val
        self.loss_matrix = np.asarray(loss, dtype="float32")
        self.exp_matrix = np.asarray(exp, dtype="float32")
        self.gain_matrix = np.asarray(gain, dtype="float32")
        self.mut_matrix = np.asarray(mut, dtype="float32")
        self.methy_matrix = np.asarray(methy, dtype="float32")

    def make_5_dim_image(self):
        image = np.stack((self.gain_matrix, self.loss_matrix, self.mut_matrix, self.exp_matrix,  self.methy_matrix))
        return image

    def make_5_dim_chr_image(self):
        image = np.stack((self.chr_gain_matrix, self.chr_loss_matrix, self.chr_mut_matrix,
                          self.chr_exp_matrix, self.chr_methy_matrix))
        return image


    def vector_of_all_features(self):
        tmp = np.append(self.gain_matrix.flatten(), self.loss_matrix.flatten())
        tmp = np.append(tmp, self.mut_matrix.flatten())
        tmp = np.append(tmp, self.exp_matrix.flatten())
        tmp = np.append(tmp, self.methy_matrix.flatten())
        return tmp

    def make_image_matrces_by_chr(self):

        n_chr = 22
        genes_on_chr1 = 3760
        loss = np.zeros((n_chr, genes_on_chr1))
        gain = np.zeros((n_chr, genes_on_chr1))
        mut = np.zeros((n_chr, genes_on_chr1))
        methy = np.zeros((n_chr, genes_on_chr1))
        exp = np.zeros((n_chr, genes_on_chr1))

        for gene in self.dict_of_cells:
            if self.dict_of_cells[gene].loss_val is not None:
                loss[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].loss_val
            if self.dict_of_cells[gene].gain_val is not None:
                gain[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].gain_val
            if self.dict_of_cells[gene].mut_val is not None:
                mut[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].mut_val
            if self.dict_of_cells[gene].exp_val is not None:
                exp[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].exp_val
            if self.dict_of_cells[gene].methy_val is not None:
                methy[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j] = self.dict_of_cells[gene].methy_val
        self.chr_loss_matrix = np.asarray(loss, dtype="float32")
        self.chr_exp_matrix = np.asarray(exp, dtype="float32")
        self.chr_gain_matrix = np.asarray(gain, dtype="float32")
        self.chr_mut_matrix = np.asarray(mut, dtype="float32")
        self.chr_methy_matrix = np.asarray(methy, dtype="float32")

    def _3channel_square(self):
        image = np.stack((self.mut_matrix, self.exp_matrix, self.methy_matrix))
        return image

    def _3channel_chr(self):
        image = np.stack((self.chr_mut_matrix, self.chr_exp_matrix, self.chr_methy_matrix))
        return image

    def _3channel_flat(self):
        tmp = np.append(self.mut_matrix.flatten(), self.exp_matrix.flatten())
        tmp = np.append(tmp, self.methy_matrix.flatten())
        return tmp


    def analyze_attribution(self, att_mat, n, type):
        mydict = {}
        for gene in self.dict_of_cells:
            mydict[gene] = att_mat[self.dict_of_cells[gene].i, self.dict_of_cells[gene].j]
        sort_dict = {k: v for k, v in sorted(mydict.items(), key=lambda item: item[1],reverse=True )}
        sort_dict = dict(itertools.islice(sort_dict.items(), n))
        df = pd.DataFrame.from_dict(sort_dict, orient='index')
        df['type'] = type
        df['gene'] = df.index
        df.reset_index(drop=True)
        return df