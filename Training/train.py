import json

import argparse
import numpy as np
import torch

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from Dataloader import TCGAImageLoader
from Models.AE_Square import AE,UnFlatten

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config_file',type=str, default='config')
args = parser.parse_args()
print(args)
if args.config_file is None:
    print("You have to provide a path to a config file")
    quit(1)
else:
    config_path = args.config_file

with open(config_path, "r") as jsonfile:
    config = json.load(jsonfile)
    print("Read successful")

LR = config['LR'] #9.900000000000001e-05
batch_size = config['batch_size']
lr_decay = config['lr_decay']  # 1e-5
weight_decay = config['weight_decay'] # 1e-5
epochs = config['epochs'] #200
start_of_lr_decrease = config['start_of_lr_decrease']#60
# Dataset Params
folder = config['folder'] #"Metastatic_data"
image_type = config['image_type']# "SquereImg"
predictor_column = config['predictor_column'] #
response_column = config['response_column'] #11

transform = transforms.Compose([transforms.ToTensor()])
dataset = TCGAImageLoader(config['meta_data'],
                          folder,
                          image_type,
                          predictor_column,
                          response_column)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
print("Train size: ", train_size)
print("Test size: ", test_size)
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

trainLoader = DataLoader(train_set, batch_size=batch_size, num_workers=10, shuffle=True)
valLoader = DataLoader(val_set, batch_size=batch_size,num_workers=10, shuffle=True)


net, cost_func = AE(output_size=2), torch.nn.CrossEntropyLoss()
cost_func_reconstruct = torch.nn.MSELoss()
net.to(device)

optimizer = torch.optim.Adagrad(net.parameters(), lr_decay=lr_decay, lr=LR, weight_decay=weight_decay)
lambda1 = lambda epoch: 0.99 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

best_loss = float("+Inf")
best_model = None
trigger_times = 0
last_loss = 0
patience = config['early_stop_patience']

def acc(y_hat, y):
    probs = y_hat
    winners = probs.argmax(dim=1)
    corrects = (winners == y)
    accuracy = corrects.sum().float() / float(y.size(0))
    return accuracy, winners

def batch_train(x, y):
    net.train()
    y_hat,recon,L = net(x)
    loss = cost_func(y_hat, y)
    cost_func.zero_grad()
    cost_func_reconstruct.zero_grad()
    accuracy, pred_classes = acc(y_hat, y)
    auc = 0
    if config['trainer'] != "multi-class":
        auc = roc_auc_score(y_true=y.cpu().detach(), y_score=pred_classes.cpu().detach())
    report = classification_report(
        digits=6,
        y_true=y.cpu().detach().numpy(),
        y_pred=pred_classes.cpu().detach().numpy(),
        output_dict=True,
        zero_division=0)
    total_loss = loss
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), accuracy.item(), report['macro avg']['precision'],report['macro avg']['recall'],report['macro avg']['f1-score'], auc

def batch_valid(x, y):
    with torch.no_grad():
        net.eval()
        y_hat,recon,L = net(x)
        loss = cost_func(y_hat, y)

        accuracy, pred_classes = acc(y_hat, y)
        auc = 0
        if config['trainer'] != "multi-class":
            auc = roc_auc_score(y_true=y.cpu().detach(), y_score=pred_classes.cpu().detach())
        report = classification_report(
            digits=6,
            y_true=y.cpu().detach().numpy(),
            y_pred=pred_classes.cpu().detach().numpy(),
            output_dict=True,
            zero_division=0)
        total_loss = loss
        return total_loss.item(), accuracy.item(), report['macro avg']['precision'],report['macro avg']['recall'],report['macro avg']['f1-score'], auc

def saveModel(ep, optimizer, loss):
    torch.save({
        'epoch': ep,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, "saved_models/best_model.pb")

train_losses = []
val_losses = []
for ep in range(epochs):
    batch_train_f1,batch_val_auc, batch_train_auc,\
    batch_train_loss, batch_val_f1, batch_val_loss = [],[],[],[],[],[]
    for x, y_dat,id in trainLoader:
        loss, acc_train, precision,recall,f1,train_auc = batch_train(x.cuda(), y_dat.cuda())
        batch_train_loss.append(loss)
        batch_train_f1.append(f1)
        batch_train_auc.append(train_auc)

    for x, y_dat,id in valLoader:
        loss, acc_val,  precision,recall,f1,val_auc = batch_valid(x.cuda(), y_dat.cuda())
        batch_val_loss.append(loss)
        batch_val_f1.append(f1)
        batch_val_auc.append(val_auc)
    if ep >= start_of_lr_decrease:
        scheduler.step()
    print(
        "Epoch {}: \n\tTrain loss: {} Train F1: {} \n\tValidation loss: {} Val F1: {}, \n\tLR : {}".format(ep,
            np.mean(batch_train_loss),np.mean(batch_train_f1),
            np.mean( batch_val_loss),np.mean(batch_val_f1),
            optimizer.param_groups[0]["lr"])
    )
    if np.mean(batch_val_loss) < best_loss:
        best_loss = np.mean(batch_val_loss)
        best_model = net
        print("Best loss! ")

    if (np.mean(batch_train_auc) >= config['save_model_score'] and
            np.mean(batch_val_auc) >= config['save_model_score']) and \
            np.mean(batch_val_loss) > last_loss:
        saveModel(ep, optimizer, np.mean(batch_val_loss))
    if np.mean(batch_val_loss) > last_loss:
        trigger_times += 1
        print('Trigger Times:', trigger_times)
        if trigger_times >= patience:
            print("Early Stopping!")
            break
    else:
        print('trigger times: 0')
        trigger_times = 0
    last_loss = np.mean(batch_val_loss)