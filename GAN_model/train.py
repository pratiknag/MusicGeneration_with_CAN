import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def train_DCGAN(G, D, optim_G, optim_D, loss_f, train_loader, num_epochs, device):

    for epoch in range(num_epochs):

        for i, (img) in enumerate(train_loader):
            batch_size = img.shape[0]

            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)

            # ========================
            #   Train Discriminator
            # ========================
            # train with real data
            img = img.to(device)
            real_score = D(img.float())
            d_loss_real = loss_f(real_score.flatten(), real_label)

            # train with fake data
            noise = torch.randn(batch_size, 100, device=device)
            img_fake = G(noise)
            fake_score = D(img_fake)
            d_loss_fake = loss_f(fake_score.flatten(), fake_label)

            # update D
            d_loss = d_loss_real + d_loss_fake
            D.zero_grad()
            d_loss.backward()
            optim_D.step()

            # ========================
            #   Train Generator
            # ========================
            noise = torch.randn(batch_size, 100, device=device)
            img_fake = G(noise)
            g_score = D(img_fake)
            g_loss = loss_f(g_score.flatten(), real_label)

            # update G
            G.zero_grad()
            g_loss.backward()
            optim_G.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item(),
                         real_score.mean().item(), fake_score.mean().item(), g_score.mean().item()))

        noise = torch.randn(2, 100, device=device)
        img_fake = G(noise)
        grid = make_grid(img_fake)
        plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
        plt.show()