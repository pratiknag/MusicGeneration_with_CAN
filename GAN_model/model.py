import torch.nn as nn


class Generator(nn.Module):
    # this generator is a simplified version of DCGAN to speed up the training

    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # input size 100 x 1 x 1
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            # size 512 x 4 x 4
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # size 128 x 16 x 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #  size 128 x 32 x 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #  size 64 x 64 x 64
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            #  output size 1 x 128 x 128
        )

    def forward(self, z):
        if z.shape[-1] != 1:
            # change the shape from (batch_size, 100) to (batch_size, 100, 1, 1)
            z = z[..., None, None]

        output = self.net(z)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # input size batch x 1 x 128 x 128
            nn.Conv2d(1, 256, 4, 2, 1, bias=False) ,
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            # input size batch x 256 
            nn.Conv2d(256, 512, 4, 2, 1, bias=False) ,
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Flatten(),
            # input size batch x 524288 
            nn.Linear(524288,1),
            nn.Sigmoid()
            
        )

    def forward(self, img):
        return self.net(img)
