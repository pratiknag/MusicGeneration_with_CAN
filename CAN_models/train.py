import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import numpy as np
import torch.nn as nn
import copy
import tqdm
#from keras.utils import to_categorical
def train_DCGAN(G, D, optim_G, optim_D, loss_f1, loss_f2, train_loader, num_epochs, label2onehot, 
            CrossEntropy_uniform, n_class, device):

    for epoch in range(num_epochs):

        for i, (img,labels) in enumerate(train_loader):
            batch_size = img.shape[0]
            #labels = label2onehot(labels)
            #labels = labels.long()
            
            real_label = torch.ones(batch_size, device=device)
            fake_label = torch.zeros(batch_size, device=device)
            fake_r_score = torch.zeros(batch_size, device=device)
            fake_c_score = torch.zeros(batch_size, device=device)
            # ========================
            #   Train Discriminator
            # ========================
            # train with real data
            img = img.to(device)
            labels = labels.to(device)
            real_r_score, real_c_score = D(img)
            

            # train with fake data
            noise = torch.randn(batch_size, 100, device=device)
            img_fake = G(noise)

            fake_r_score, fake_c_score = D(img_fake)
            

            # update D      
            #print(np.resize(labels,(labels.length())).shape)
       
            d_loss = loss_f1(real_r_score.flatten(), real_label) + loss_f2(real_c_score, labels)
            d_loss = d_loss + loss_f1(fake_r_score.flatten(), fake_label)
            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()
                    
            # ========================
            #   Train Generator
            # ========================
            
            
            # update G
            optim_G.zero_grad()
            unrolled_steps=1
            if unrolled_steps > 0:
                backup = copy.deepcopy(D)
            for j in range(unrolled_steps):
                print('.')
                img = img.to(device)
                labels = labels.to(device)
                real_r_score, real_c_score = D(img)


                # train with fake data
                noise = torch.randn(batch_size, 100, device=device)
                img_fake = G(noise)

                fake_r_score, fake_c_score = D(img_fake)


                # update D      
                #print(np.resize(labels,(labels.length())).shape)

                d_loss = loss_f1(real_r_score.flatten(), real_label) + loss_f2(real_c_score, labels)
                d_loss = d_loss + loss_f1(fake_r_score.flatten(), fake_label)
                optim_D.zero_grad()
                d_loss.backward(create_graph=True)
                optim_D.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

            img_fake = G(noise)
            fake_r_score_2, fake_c_score_2 = D(img_fake)
            logsoftmax = nn.LogSoftmax(dim=1)
            unif = torch.full((batch_size, n_class), 1/n_class)
            unif = unif.to(device)
            # Calculate G's loss based on this output
            g_loss = loss_f1(fake_r_score_2.flatten(), real_label)
            g_loss = g_loss + torch.mean(-torch.sum(unif * logsoftmax(fake_c_score_2 ), 1))
            # Calculate gradients for G
            g_loss.backward()
            optim_G.step()
            if unrolled_steps > 0:
                D.load(backup)    
                del backup
            if i % 5 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_loader), d_loss.item(), g_loss.item(),
                         real_r_score.mean().item(), fake_r_score.mean().item(), fake_r_score_2.mean().item()))

                noise = torch.randn(2, 100, device=device)
                img_fake = G(noise)
                grid = make_grid(img_fake)
                plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
                plt.show()
        
        
def CrossEntropy_uniform(self, pred):
    logsoftmax = nn.LogSoftmax(dim=1).to(device)
    unif = torch.full((self.batch_size, self.n_class), 1/self.n_class).to(device)
    return torch.mean(-torch.sum(unif * logsoftmax(pred), 1)).to(device)
    
    