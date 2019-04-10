#%%
import torch
import visdom
import numpy as np
import time
#%%
device = torch.device("cpu")
checkpoint = torch.load("models/10-04-19-model_new", map_location = device)

#%%
losses = checkpoint["loss"]
accuracies = checkpoint["accuracy"]

plot_val_to_visdom(losses['train'][1:], losses['valid'][1:], "Loss")
plot_val_to_visdom(accuracies['train'], accuracies['valid'], "Accuracy")

#%%
def plot_val_to_visdom(train_val, valid_val, val_name):
    n = len(train_val)
    epochs = np.arange(n)

    vis = visdom.Visdom()
    win = vis.line(
        X = np.column_stack((epochs, epochs)),
        Y = np.column_stack((train_val, valid_val)),
        opts = dict(
            legend = ["Train", "Valid"],
            title = val_name,
            xlabel = "Epochs",
            ylabel = val_name
        )
    )      