import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
from model import *
import matplotlib.pyplot as plt
from topic_tree import  plot_tree



class GBN_trainer:
    def __init__(self, args, voc_path='voc.txt'):
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.save_path = args.save_path
        self.epochs = args.epochs
        self.voc = self.get_voc(voc_path)
        self.layer_num = len(args.topic_size)
        self.model = GBN_model(args)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                                  lr=self.lr, weight_decay=self.weight_decay)

    def train(self, train_data_loader, test_data_loader):

        for t in range(self.layer_num - 1, -1, 1):
            self.decoder[t - 1].alphas = self.decoder[t].rho

        for epoch in range(self.epochs):

            # for t in range(self.layer_num - 1):
            #     self.model.decoder[t + 1].rho = self.model.decoder[t].alphas
            self.model.cuda()

            for i, (train_data, _) in enumerate(train_data_loader):
                 # for step in range(self.layer_num):

                    loss, likelihood, KL, graph_lh = self.model([ torch.tensor(train_data[0], dtype=torch.float).cuda(),  torch.tensor(train_data[1], dtype=torch.float).cuda(), torch.tensor(train_data[0], dtype=torch.float).cuda(),
                                                        torch.tensor(train_data[2],dtype=torch.float).cuda(), torch.tensor(train_data[3], dtype=torch.float).cuda(),
                                                        torch.tensor(train_data[4], dtype=torch.float).cuda() ])

                    Q_value = torch.tensor(0.).cuda()
                    for t in range(self.layer_num - 1, -1, -1):  # from layer layer_num-1-step to 0
                        Q_value += loss[t]
                    Q_value.backward()

                    for para in self.model.parameters():
                        flag = torch.sum(torch.isnan(para))

                    if (flag == 0):
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100, norm_type=2)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

            if epoch % 5 == 0:
                for t in range(self.layer_num):
                     print('epoch {}|{}, layer: {}, loss: {}, likelihood: {}, graph lh: {}, KL: {}'.format(epoch, self.epochs,t, loss[t].item(),  likelihood[t].item(), graph_lh[t].item(), KL[t].item()))
                self.vis_txt(self.save_path)


            if epoch % 5 == 0:
                self.model.eval()
                save_checkpoint({'state_dict': self.model.state_dict(), 'epoch': epoch, "rho":self.model.rho}, self.save_path, True)
                print('epoch {}|{}, test_ikelihood,{}'.format(epoch, self.epochs, self.test(train_data_loader)))



    def test(self, data_loader):
        self.model.eval()
        likelihood_t = 0
        num_data = len(data_loader)
        ppl_total = 0

        for i, (train_data, test_data) in enumerate(data_loader):
            train_data = torch.tensor(train_data[0], dtype=torch.float).cuda()
            test_data = torch.tensor(test_data, dtype=torch.float).cuda()
            # test_label = torch.tensor(test_label, dtype=torch.long).cuda()

            with torch.no_grad():
                ppl = self.model.test_ppl(train_data, test_data)
                # likelihood_total += ret_dict["likelihood"][0].item() / num_data
                ppl_total += ppl.item() / num_data

            # re_x, theta, loss_list, likelihood = self.model(test_data)
            # likelihood_t += likelihood[0].item() / num_data

        # save_checkpoint({'state_dict': self.model.state_dict(), 'epoch': epoch}, self.save_path, True)
        return ppl_total


    def load_model(self):
        checkpoint = torch.load(self.save_path)
        self.GBN_models.load_state_dict(checkpoint['state_dict'])

    def vis(self):
        # layer1
        w_1 = torch.mm(self.GBN_models[0].decoder.rho, torch.transpose(self.GBN_models[0].decoder.alphas, 0, 1))
        phi_1 = torch.softmax(w_1, dim=0).cpu().detach().numpy()

        index1 = range(100)
        dic1 = phi_1[:, index1[0:49]]
        # dic1 = phi_1[:, :]
        fig7 = plt.figure(figsize=(10, 10))
        for i in range(dic1.shape[1]):
            tmp = dic1[:, i].reshape(28, 28)
            ax = fig7.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index1[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_2 = torch.mm(self.GBN_models[1].decoder.rho, torch.transpose(self.GBN_models[1].decoder.alphas, 0, 1))
        phi_2 = torch.softmax(w_2, dim=0).cpu().detach().numpy()
        index2 = range(49)
        dic2 = np.matmul(phi_1, phi_2[:, index2[0:49]])
        #dic2 = np.matmul(phi_1, phi_2[:, :])
        fig8 = plt.figure(figsize=(10, 10))
        for i in range(dic2.shape[1]):
            tmp = dic2[:, i].reshape(28, 28)
            ax = fig8.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index2[i] + 1))
            ax.imshow(tmp)

        # layer2
        w_3 = torch.mm(self.GBN_models[2].decoder.rho, torch.transpose(self.GBN_models[2].decoder.alphas, 0, 1))
        phi_3 = torch.softmax(w_3, dim=0).cpu().detach().numpy()
        index3 = range(32)

        dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, index3[0:32]])
        #dic3 = np.matmul(np.matmul(phi_1, phi_2), phi_3[:, :])
        fig9 = plt.figure(figsize=(10, 10))
        for i in range(dic3.shape[1]):
            tmp = dic3[:, i].reshape(28, 28)
            ax = fig9.add_subplot(7, 7, i + 1)
            ax.axis('off')
            ax.set_title(str(index3[i] + 1))
            ax.imshow(tmp)

        plt.show()

    def get_voc(self, voc_path):
        if type(voc_path) == 'str':
            voc = []
            with open(voc_path) as f:
                lines = f.readlines()
            for line in lines:
                voc.append(line.strip())
            return voc
        else:
            return voc_path

    def vision_phi(self, Phi, outpath='phi_output_1', top_n=50):
        if self.voc is not None:
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            phi = 1
            for num, phi_layer in enumerate(Phi):
                phi = phi_layer
                phi_k = phi.shape[1]
                path = os.path.join(outpath, 'phi' + str(num) + '.txt')
                f = open(path, 'w')
                for each in range(phi_k):
                    top_n_words = self.get_top_n(phi[:, each], top_n)
                    f.write(top_n_words)
                    f.write('\n')
                f.close()
        else:
            print('voc need !!')

    def get_top_n(self, phi, top_n):
        top_n_words = ''
        idx = np.argsort(-phi)
        for i in range(top_n):
            index = idx[i]
            top_n_words += self.voc[index]
            top_n_words += ' '
        return top_n_words

    def vis_txt(self, path):
        phi = []
        for t in range(self.layer_num):
            # w_t = torch.mm(self.model.decoder[t].rho, torch.transpose(self.model.decoder[t].alphas, 0, 1))
            # w_t = torch.mm(self.model.decoder, torch.transpose(self.model.decoder[t].alphas, 0, 1))
            w_t = torch.mm(self.model.rho[t], torch.transpose(self.model.topic_embedding[t + 1](), 0, 1))
            phi_t = torch.softmax(w_t, dim=0).cpu().detach().numpy()
            phi.append(phi_t)
        # for t in range (16):
        #     graph = plot_tree(phi, self.voc, topic_id=t, threshold=0.05)
        #     graph.render(filename='pic' + str(t), directory='phi_output', view=False)
        self.vision_phi(phi, outpath=path + '_phi_output')