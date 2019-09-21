__author__ = "Soumyakanti Das"

"""
This file contains code that implements Adversarial Autoencoder to generate
images similar to MNIST dataset.
"""

import torch
import torchvision
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader as DataLoader

# transform into a tensor.
transform = transforms.Compose([
    transforms.ToTensor()
#     transforms.Normalize([0.5], [0.5])
])

batch_size = 256

training_set = datasets.MNIST("./MNIST", train=True, transform=transform, download=True)
train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

inp_dim = 784
N = 1000
z_dim = 2
epochs = 100
variance = 5

#Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.lin1 = nn.Linear(inp_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, z_dim)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin3(x)
        return x

#Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, inp_dim)
    def forward(self, x):
        x = self.lin1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin3(x)
        return torch.sigmoid(x)
#         return x

#Discriminator
class Discrim(nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        # output dimension is 1 because it returns a probability
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = F.dropout(self.lin1(x), p=0.2, training=self.training)
        x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.2, training=self.training)
        x = F.relu(x)
        return torch.sigmoid(self.lin3(x))

# set device for cuda optimization.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder()
encoder.to(device)

decoder = Decoder()
decoder.to(device)

discrim = Discrim()
discrim.to(device)

# used for reconstruction loss
recon_criterion = F.binary_cross_entropy

# reconstruction & Regularization learning rates
rec_lr, reg_lr = 0.0005, 0.0001

# optimizers
encoder_optim = optim.Adam(encoder.parameters(), lr=rec_lr)
decoder_optim = optim.Adam(decoder.parameters(), lr=rec_lr)
generator_optim = optim.Adam(encoder.parameters(), lr=reg_lr)
discrim_optim = optim.Adam(discrim.parameters(), lr=reg_lr)

def train():
    """
    Train all three networks for certain epochs.

    :return: list of losses.
    """
    # training mode for all networks
    encoder.train()
    decoder.train()
    discrim.train()

    # list of losses to be returned
    recon_losses = []
    discrim_losses = []
    gen_losses = []

    for epoch in range(epochs):
        # Running losses per epoch.
        recon_running = 0
        discrim_running = 0
        generator_running = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.view(-1, inp_dim)
            inputs, labels = inputs.to(device), labels.to(device)

            # gradients to zero.
            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            generator_optim.zero_grad()
            discrim_optim.zero_grad()

            #Reconstruction phase
            z = encoder(inputs)
            gen_outputs = decoder(z)
            recon_loss = recon_criterion(gen_outputs+1e-10, inputs+1e-10)
            recon_running += recon_loss.item()

            recon_loss.backward()
            encoder_optim.step()
            decoder_optim.step()

            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            generator_optim.zero_grad()
            discrim_optim.zero_grad()

            #Regularization phase

            #Discriminator
            # encoder in eval mode.
            encoder.eval()
            # generate data from gaussian distribution with some variance.
            z_real = torch.randn(inputs.shape[0], z_dim, device=device) * variance
            z_real.to(device)
            z_fake = encoder(inputs)
            gauss_z_fake = discrim(z_fake)
            gauss_z_real = discrim(z_real)
            discrim_loss = -torch.mean(torch.log(gauss_z_real + 1e-10) + torch.log(1 - gauss_z_fake + 1e-10))
            discrim_loss.to(device)
            discrim_running += discrim_loss.item()

            discrim_loss.backward()
            discrim_optim.step()

            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            generator_optim.zero_grad()
            discrim_optim.zero_grad()

            #Generator
            encoder.train()
            z_fake = encoder(inputs)
            generator_loss = -torch.mean(torch.log(discrim(z_fake) + 1e-10))
            generator_running += generator_loss.item()

            generator_loss.backward()
            generator_optim.step()

        # append losses
        recon_losses.append(recon_running/(len(training_set)/256))
        discrim_losses.append(discrim_running/(len(training_set)/256))
        gen_losses.append(generator_running/(len(training_set)/256))
        # print losses.
        print("epochs: {}, recon_loss: {}, discrim_loss: {}, gen_loss: {}". \
              format(epoch, recon_losses[-1], discrim_losses[-1], gen_losses[-1]))

    return recon_losses, discrim_losses, gen_losses

def plot(losses):
    """
    Plots losses.

    :param losses: tuple of losses.
    :return: None
    """
    r, d, g = losses

    xs = np.linspace(0, epochs, epochs)
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    axes[0].plot(xs, r)
    axes[0].set_xlabel("epochs")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Reconstruction loss")

    axes[1].plot(xs, d)
    axes[1].set_xlabel("epochs")
    axes[1].set_ylabel("loss")
    axes[1].set_title("Discriminator loss")

    axes[2].plot(xs, g)
    axes[2].set_xlabel("epochs")
    axes[2].set_ylabel("loss")
    axes[2].set_title("Generator loss")

    fig.tight_layout()

def plot_results(n=10):
    """
    Draws generated images from random normal distribution.

    :param n: number of rows and columns. Default is 10
    """
    z_real = torch.randn(n, n, z_dim, device=device)*variance
    z_real = z_real.to(torch.device("cpu"))
    with torch.no_grad():
        decoder.to(torch.device("cpu"))
        decoder.eval()
        gen = decoder(z_real)
        array = np.array(gen.view(-1, 28))
        begin = 0
        result = None

        # stacks the array into a square.
        for end in range(n*28, array.shape[0]+1, n*28):
            if begin == 0:
                result = array[begin:end]
            else:
                result = np.hstack((result, array[begin:end]))
            begin = end
        plt.figure(figsize=(12,12))
        plt.imshow(result)
        plt.axis("off")

def main():
    losses = train()
    plot(losses)
    plot_results()

if __name__ == '__main__':
    main()
