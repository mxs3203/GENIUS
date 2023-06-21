
class ImageCell:
    def __init__(self, gene, loss_val, gain_val, mut_val, exp_val, chr, methy_val ):
        self.gene = gene
        self.loss_val = loss_val
        self.gain_val = gain_val
        self.mut_val = mut_val
        self.methy_val = methy_val
        self.exp_val = exp_val
        self.chr = chr
        self.i = -1
        self.j = -1

