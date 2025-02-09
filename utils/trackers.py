import os
import errno
import numpy as np
from numpy import save, savetxt
from matplotlib.pylab import plt
from matplotlib.pyplot import text

from pytorch_lightning.callbacks import Callback

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, euclidean_distances
from scipy.stats import spearmanr, entropy
from scipy.special import softmax

import torch
import torch.nn as nn

def create_dir(path):
    ## criando o diretorio
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    # else:
    #     print("Directory already exists: " + path)

class MetricTracker(Callback):

    def __init__(self, args):
        # super(MetricTracker, self).__init__()
        self.train_loss_collection = []
        self.val_loss_collection = []
        self.train_metric_collection = []
        self.val_metric_collection = []
        self.N = 10
        self.save_dir = os.path.join(args.savedir, 'tracker') 
        self.args = args
        # time
        self.time_compute_batch = []
        
        ## distance
        self.pdist = nn.PairwiseDistance(p=2)
        self.build_anchors(args)
        
    def on_train_epoch_end(self, trainer, module):
        cb_metrics = trainer.callback_metrics
        # curve loss and metric
        self.train_loss_collection.append(cb_metrics['train_loss'].detach().cpu().numpy()) # track them
        self.train_metric_collection.append(cb_metrics['train_jac_epoch'].detach().cpu().numpy()) # track them
        self.val_loss_collection.append(cb_metrics['val_loss'].detach().cpu().numpy()) # track them
        self.val_metric_collection.append(cb_metrics['val_jac_epoch'].detach().cpu().numpy()) # track them
        # get time compute batch
        self.time_compute_batch.append(cb_metrics['total_time'].detach().cpu().numpy()) # track them
    
    # def on_validation_epoch_end(self, trainer, module):
    #     self.trainer = trainer # get values
    #     #lightning_module
    #     # get predictions X, y and categories
    #     # poderia ser colocado na validação
    #     X = trainer.lightning_module.last_X
    #     y = trainer.lightning_module.last_y 
    #     categories = trainer.lightning_module.categories
    #     args = trainer.lightning_module.args
        
    #     if args.dataset == 'geometryv2':
    #         self.plot_embeddings_2D(args, X, y, categories)
    #         self.plot_embeddings3D(args, X, y, categories)
            
    #     if args.dataset == 'vaihingen': 
    #         self.plot_all_2D(args, X, y, categories)
            
    #     # compute the distance to anchors
    #     self.save_distance_to_anchors(args, X, y, categories, folder_name='hist_epoch', idx=self.trainer.current_epoch)
    
    # def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
    #     self.trainer = trainer # get values
    #     #lightning_module
    #     # get predictions X, y and categories
    #     # poderia ser colocado na validação
    #     X = trainer.lightning_module.last_X
    #     y = trainer.lightning_module.last_y 
    #     categories = trainer.lightning_module.categories
    #     args = trainer.lightning_module.args
        
    #     if args.dataset == 'geometryv2' and self.trainer.current_epoch == 0:
    #         create_dir(os.path.join(self.args.savedir, 'batch'))
    #         self.plot_batch_embeddings3D(args, X, y, categories, batch_idx)
            
    #     if self.trainer.current_epoch == 0: 
    #         self.save_distance_to_anchors(args, X, y, categories, folder_name='hist_batch', idx=batch_idx)
         
    def on_train_end(self, trainer, module):
        # print(self.loss_collection)
        # print(self.metric_collection)
        create_dir(self.save_dir)
        save(os.path.join(self.save_dir, 'train_loss_collection.npy'), self.train_loss_collection)
        savetxt(os.path.join(self.save_dir, 'train_loss_collection.csv'), self.train_loss_collection, delimiter=';', fmt='%0.4f')
        
        save(os.path.join(self.save_dir, 'train_metric_collection.npy'), self.train_metric_collection)
        savetxt(os.path.join(self.save_dir, 'train_metric_collection.csv'), self.train_metric_collection, delimiter=';', fmt='%0.4f')
        
        save(os.path.join(self.save_dir, 'val_loss_collection.npy'), self.val_loss_collection)
        savetxt(os.path.join(self.save_dir, 'val_loss_collection.csv'), self.val_loss_collection, delimiter=';', fmt='%0.4f')
        
        save(os.path.join(self.save_dir, 'val_metric_collection.npy'), self.val_metric_collection)
        savetxt(os.path.join(self.save_dir, 'val_metric_collection.csv'), self.val_metric_collection, delimiter=';', fmt='%0.4f')
        
        # save plot loss and metric curve
        self.plot_curve(self.train_loss_collection, 'Train Loss', os.path.join(self.save_dir, 'train_loss_curve.png'))
        self.plot_curve(self.train_metric_collection, 'Train mIoU', os.path.join(self.save_dir, 'train_miou_curve.png'))
        self.plot_curve(self.val_loss_collection, 'Val Loss', os.path.join(self.save_dir, 'val_loss_curve.png'))
        self.plot_curve(self.val_metric_collection, 'Val mIoU', os.path.join(self.save_dir, 'val_miou_curve.png'))
        
        if self.args.loss == 'multiple_triplet':
            save(os.path.join(self.save_dir, 'k_prototypes.npy'), module.k_prototypes.detach().cpu().numpy())

        # save time
        save(os.path.join(self.save_dir, 'time_compute_batch.npy'), self.time_compute_batch)
        savetxt(os.path.join(self.save_dir, 'time_compute_batch.csv'), self.time_compute_batch, delimiter=';', fmt='%0.4f')
        
        f = open(os.path.join(self.save_dir, 'time_compute_batch.txt'), 'w')
        mean_seconds = np.mean(self.time_compute_batch)
        f.write("Mean seconds ..........:" + str(mean_seconds))
        f.write("\nFormated mean seconds .:" + str(round(mean_seconds, 3)))
        f.write("\n ---------")
        m, s = divmod(mean_seconds, 60)
        f.write("\nSeconds.:" + str(s))
        f.write("\nMinutes.:" + str(m))
        f.write("\nFormated seconds.:" + str(f'{s:.3f}'))
        f.write("\nFormated minutes.:" + str(f'{m:.0f}'))
        f.close()   
    
    #####################################################################################################################    
    def plot_curve(self, values, label, file_path):
        
        plt.clf()
        
        # Generate a sequence of integers to represent the epoch numbers
        epochs = range(1, len(values)+1)
        
        # Plot and label the training and validation loss values
        plt.plot(epochs, values, label=label)

        # Add in a title and axes labels
        # plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        # Set the tick locations
        # plt.xticks(arange(0, len(values), 2))
        
        # Display the plot
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(file_path)
    
    def select_n_emb_per_class(self, X, y, n):
        
        X_temp, y_temp = [], []
        for c in np.unique(y):
            tmp_x = X[y == c]
            tmp_y = y[y == c]
            indices = np.random.randint(tmp_x.shape[0], size=n)
            tmp_x = tmp_x[indices]
            tmp_y = tmp_y[indices]
            X_temp.append(tmp_x)
            y_temp.append(tmp_y)
        
        X_temp = np.concatenate(X_temp)
        y_temp = np.concatenate(y_temp)
        
        return X_temp, y_temp
    
    def plot_embeddings_2D(self, args, X, y, categories):

        # resize torch tensor
        # torch to numpy
        X = X.permute(0, 2, 3, 1).contiguous()
        shape = X.size()
        X = X.view(shape[0] * shape[1] * shape[2], shape[3])
        X = X.detach().cpu().numpy()
        
        y = y.view(shape[0] * shape[1] * shape[2])
        y = y.detach().cpu().numpy()

        # print(X.shape)

        # remove ignore index
        mask = np.ones(len(y), dtype=bool)
        mask[y == 255] = False
        X = X[mask,...]
        y = y[mask,...]
        
        # random get indx
        X, y = self.select_n_emb_per_class(X, y, 5000)
        
        # save silhouette_score
        # % O melhor valor é 1 e o pior valor é -1. Valores próximos a 0 indicam clusters sobrepostos.
        ss = silhouette_score(X=X, labels=y)
        dbs = davies_bouldin_score(X=X, labels=y)
        chs = calinski_harabasz_score(X=X, labels=y)
        
        file_name = os.path.join(args.savedir, 'tsne_epoch_'+ str(self.trainer.current_epoch) + '_cluster_score.txt')
        f = open(file_name, 'w')
        
        f.write("--- cluster evaluation --- \n")
        f.write("silhouette_score.: " + str(ss) + '\n')
        f.write("davies_bouldin_score.: " + str(dbs) + '\n')
        f.write("calinski_harabasz_score.: " + str(chs) + '\n')
        
        f.close()
        
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9770772
        self.cosine_similarity_index(args, X, y, categories)
        self.cosine_distance_index(args, X, y, categories)
        self.euclidean_distance_index(args, X, y, categories)
        self.pearson_correlation_index(args, X, y, categories)
        # self.spearman_correlation_index(args, X, y, categories)
        # media a incerteza ou confidencia de cada saída softmax?
        # https://towardsdatascience.com/2-easy-ways-to-measure-your-image-classification-models-uncertainty-1c489fefaec8
        # https://towardsdatascience.com/knowing-known-unknowns-with-deep-neural-networks-caac1c4c1f5d
        # Predictive uncertainty can combine epistemic and aleatoric uncertainty
        # Um modelo pode ter baixos valores de ativação em todos os neurônios de sua camada de saída e ainda assim chegar a um alto valor de softmax
        # Monte Carlo Dropout e Deep Ensembles
        # https://github.com/RobRomijnders/bayes_nn
        # entropia sobre as saídas do softmax (para estimar as incertezas)
        # variance softmax entropy
        self.uncertainty_index(args, X, y, categories)
        
        # save avg logits
        self.save_means_logits(args, X, y, categories)
        
        # compute the TSNE
        tsne = TSNE(n_components=2, learning_rate='auto', init='pca', n_iter=1000, perplexity=50, verbose=0)
        X_hat = tsne.fit_transform(X) # returns shape (n_samples, 2)
        
        # dict
        cdict = {0:'#333333', 1:'#0343ff', 2:'#15b01a'}
        labl = dict(zip(range(3), categories)) 
        marker = {0:'*',1:'s',2:'o'}
        alpha = {0:.3, 1:.5, 2:.7}
        
        fig = plt.figure(figsize=(7,5))
        fig.patch.set_facecolor('white')
        
        for i in np.unique(y):
            mask = np.where(y == i)
            X_label = X_hat[mask]
            # plt.scatter(X_label[:, 0], X_label[:, 1], label=i)
            plt.scatter(X_label[:, 0], X_label[:, 1], c=cdict[i], s=40,
                label=labl[i], marker=marker[i], alpha=alpha[i])
            
        # plt.legend()
        plt.legend(loc='best', markerscale=2, fontsize=10)
        
        # plt.show()
        file_path = os.path.join(args.savedir,'tsne_epoch_'+ str(self.trainer.current_epoch)+'.png')
        plt.savefig(file_path)
           
    def plot_all_2D(self, args, X, y, categories):

        # resize torch tensor
        # torch to numpy
        X = X.permute(0, 2, 3, 1).contiguous()
        shape = X.size()
        X = X.view(shape[0] * shape[1] * shape[2], shape[3])
        X = X.detach().cpu().numpy()
        
        y = y.view(shape[0] * shape[1] * shape[2])
        y = y.detach().cpu().numpy()

        # print(X.shape)

        # remove ignore index
        mask = np.ones(len(y), dtype=bool)
        mask[y == 255] = False
        X = X[mask,...]
        y = y[mask,...]
        
        # random get indx
        X, y = self.select_n_emb_per_class(X, y, 5000)
        
        # compute 
        # save silhouette_score
        # % O melhor valor é 1 e o pior valor é -1. Valores próximos a 0 indicam clusters sobrepostos.
        ss = silhouette_score(X=X, labels=y)
        dbs = davies_bouldin_score(X=X, labels=y)
        chs = calinski_harabasz_score(X=X, labels=y)
        
        file_name = os.path.join(args.savedir, 'tsne_epoch_'+ str(self.trainer.current_epoch) + '_cluster_score.txt')
        f = open(file_name, 'w')
        f.write("--- cluster evaluation --- \n")
        f.write("silhouette_score.: " + str(ss) + '\n')
        f.write("davies_bouldin_score.: " + str(dbs) + '\n')
        f.write("calinski_harabasz_score.: " + str(chs) + '\n')
        f.close()
        
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9770772
        self.cosine_similarity_index(args, X, y, categories)
        self.cosine_distance_index(args, X, y, categories)
        self.euclidean_distance_index(args, X, y, categories)
        self.pearson_correlation_index(args, X, y, categories)
        # self.spearman_correlation_index(args, X, y, categories)
        self.uncertainty_index(args, X, y, categories)
        
        # random get indx
        # indices = np.random.randint(X.shape[0], size=10000)
        # X = X[indices]
        # y = y[indices]
        
        # compute the TSNE
        tsne = TSNE(n_components=2, learning_rate=200, init='pca', n_iter=1000, perplexity=50, verbose=0)
        X_hat = tsne.fit_transform(X) # returns shape (n_samples, 2)
        
        # dict
        labl = dict(zip(range(len(categories)), categories)) 
        fig = plt.figure(figsize=(7,5))
        fig.patch.set_facecolor('white')
        
        for i in np.unique(y):
            mask = np.where(y == i)
            X_label = X_hat[mask]
            # plt.scatter(X_label[:, 0], X_label[:, 1], label=i)
            plt.scatter(X_label[:, 0], X_label[:, 1], s=40, label=labl[i])
            
        # plt.legend()
        plt.legend(loc='best', markerscale=2, fontsize=10)
        
        # plt.show()
        file_path = os.path.join(args.savedir,'tsne_epoch_'+ str(self.trainer.current_epoch) + '.png')
        plt.savefig(file_path)
        
    def uncertainty_index(self, args, X, y, categories):
        # https://towardsdatascience.com/entropy-is-a-measure-of-uncertainty-e2c000301c2c#:~:text=Entropy%20allows%20us%20to%20make,is%20a%20measure%20of%20uncertainty%20.
        # https://github.com/RobRomijnders/weight_uncertainty
        # https://arxiv.org/pdf/1711.08244.pdf
        
        file_name = os.path.join(args.savedir, 'tsne_epoch_'+ str(self.trainer.current_epoch) + '_uncertainty.txt')
        f = open(file_name, 'w')
        f.write("--- Uncertainty estimation --- \n")
        # high values more uncertainly more information
        
        # embeddings to softmax probability
        x_softmax = softmax(X, axis=1)
        
        # for i in np.unique(y):
        #     index_class = np.where(y==i)
        #     var = np.var(x_softmax[index_class]) 
        #     f.write("variance softmax -" + str(categories[i]) + ".: " + str(var) + '\n')

        # # overall
        # f.write("variance softmax - overall.: " + str(np.var(x_softmax)) + '\n')
        
        # for i in np.unique(y):
        #     index_class = np.where(y==i)
        #     var = np.var(X[index_class]) 
        #     f.write("variance embeddings -" + str(categories[i]) + ".: " + str(var) + '\n')
            
        # # overall
        # f.write("variance embeddings - overall.: " + str(np.var(X)) + '\n')
        
        # average entropy 
        for i in np.unique(y):
            index_class = np.where(y==i)
            e_entropy = entropy(x_softmax[index_class], axis=1, base=2)
            e_ = np.mean(e_entropy)
            f.write("average entropy -" + str(categories[i]) + ".: " + str(e_) + '\n')
            
        # overall
        f.write("average entropy - overall.: " + str(np.mean(entropy(x_softmax, axis=1, base=2))) + '\n')
        
        # prediction is the average of all the sampled predictions
        # predictive entropy 
        all_mean = [] 
        for i in np.unique(y):
            index_class = np.where(y==i)
            m_softmax = np.mean(x_softmax[index_class], axis=0) 
            prediction_entropy = entropy(m_softmax, base=2)
            f.write("predictive entropy -" + str(categories[i]) + ".: " + str(prediction_entropy) + '\n')
            all_mean.append(prediction_entropy)
            
        # overall
        mean_overall = np.sum(all_mean) / len(np.unique(y))
        f.write("predictive entropy - overall.: " + str(mean_overall) + '\n')
            
        f.close()
    
    def save_means_logits(self, args, X, y, categories):
        # https://towardsdatascience.com/entropy-is-a-measure-of-uncertainty-e2c000301c2c#:~:text=Entropy%20allows%20us%20to%20make,is%20a%20measure%20of%20uncertainty%20.
        # https://github.com/RobRomijnders/weight_uncertainty
        # https://arxiv.org/pdf/1711.08244.pdf
        
        file_name = os.path.join(args.savedir, 'softmax_epoch_'+ str(self.trainer.current_epoch) + '_mean.txt')
        f = open(file_name, 'w')
        f.write("--- Softmax mean --- \n")
        # high values more uncertainly more information
        
        # embeddings to softmax probability
        x_softmax = softmax(X, axis=1)
        
        # average entropy 
        for i in np.unique(y):
            index_class = np.where(y==i)
            means = np.mean(x_softmax[index_class], axis=0)
            f.write(str(categories[i]) + ".: " + str(means) + '\n')
            
        f.close()
        
    # compute the distance to anchors
    def build_anchors(self, args):
        self.anchors = torch.zeros((args.num_classes, args.num_classes), device='cuda')
        self.magnitude = args.magnitude
        for i in range(args.num_classes): # num_classes
            self.anchors[i][i] = self.magnitude
    
    def save_distance_to_anchors(self, args, X, y, categories, folder_name, idx):
        
        # resize torch tensor
        # torch to numpy
        X = X.permute(0, 2, 3, 1).contiguous()
        shape = X.size()
        X = X.view(shape[0] * shape[1] * shape[2], shape[3])
        # X = X.detach().cpu().numpy()
        
        y = y.view(shape[0] * shape[1] * shape[2])
        y = y.detach().cpu().numpy()

        # print(X.shape)

        # remove ignore index
        mask = np.ones(len(y), dtype=bool)
        mask[y == 255] = False
        X = X[mask,...]
        y = y[mask,...]
        
        create_dir(os.path.join(self.args.savedir, folder_name))
        file_name = os.path.join(args.savedir, folder_name, 'distances_anchors_'+ str(idx) + '.txt')
        f = open(file_name, 'w')
        f.write("--- euclidean distances to anchors --- \n")
        
        # average entropy 
        for i in np.unique(y):
            index_class = np.where(y==i)
            embeddings = X[index_class].clone()
            dm = self.pdist(embeddings, self.anchors[i])
            mean = torch.mean(dm)
            std = torch.std(dm)
            f.write(str(categories[i]) + " (mean) " + ".: " + str(mean) + '\n')
            f.write(str(categories[i]) + " (std ) " + ".: " + str(std) + '\n')

        f.write('----------------------------------------------------------\n')
        f.close()
        
        # save histograma
        # plt.legend(categories) 
        plt.cla()
        plt.clf()
        for i in np.unique(y):
            embeddings = X[index_class].clone()
            dm = self.pdist(embeddings, self.anchors[i])
            plt.hist(dm.detach().cpu().numpy())
        
            # plt.legend(categories)   
            # plt.savefig(os.path.join(self.args.savedir, 'hist', 'hist_' + str(categories[i]) + '_' + str(self.trainer.current_epoch) + '_.png'))
        plt.legend(categories)   
        plt.savefig(os.path.join(self.args.savedir, folder_name, 'hist_' + str(idx) + '_.png'))
        
    def cosine_similarity_index(self, args, X, y, categories):
        
        file_name = os.path.join(args.savedir, 'tsne_epoch_'+ str(self.trainer.current_epoch) + '_cosine_similarity.txt')
        f = open(file_name, 'w')
        f.write("--- Embedding evaluation --- \n")
        
        all_mean = [] 
        for i in np.unique(y):
            index_class = np.where(y==i)
            m = cosine_similarity(X=X[index_class])
            # calculating the off-diagonal elements
            np.fill_diagonal(m, -np.inf)
            size = len(m)
            m = m.reshape(size * size)
            m = m[m != -np.inf]
            mean_value = np.mean(m)
            all_mean.append(mean_value)
            
            f.write("cosine_similarity -" + str(categories[i]) + ".: " + str(mean_value) + '\n')
            
        # overall
        mean_overall = np.sum(all_mean) / len(np.unique(y))
        f.write("cosine_similarity - overall.: " + str(mean_overall) + '\n')    
        f.close() 
        
    def cosine_distance_index(self, args, X, y, categories):
        
        file_name = os.path.join(args.savedir, 'tsne_epoch_'+ str(self.trainer.current_epoch) + '_cosine_distance.txt')
        f = open(file_name, 'w')
        f.write("--- Embedding evaluation --- \n")
        
        all_mean = [] 
        for i in np.unique(y):
            index_class = np.where(y==i)
            m = cosine_distances(X=X[index_class])
            # calculating the off-diagonal elements
            np.fill_diagonal(m, -np.inf)
            size = len(m)
            m = m.reshape(size * size)
            m = m[m != -np.inf]
            mean_value = np.mean(m)
            all_mean.append(mean_value)
            
            f.write("cosine_distance -" + str(categories[i]) + ".: " + str(mean_value) + '\n')
            
        # overall
        mean_overall = np.sum(all_mean) / len(np.unique(y))
        f.write("cosine_distances - overall.: " + str(mean_overall) + '\n')    
        f.close()
        
    def euclidean_distance_index(self, args, X, y, categories):
        
        file_name = os.path.join(args.savedir, 'tsne_epoch_'+ str(self.trainer.current_epoch) + '_euclidean_distance.txt')
        f = open(file_name, 'w')
        f.write("--- Embedding evaluation --- \n")
        
        all_mean = [] 
        for i in np.unique(y):
            index_class = np.where(y==i)
            m = euclidean_distances(X=X[index_class])
            # calculating the off-diagonal elements
            np.fill_diagonal(m, -np.inf)
            size = len(m)
            m = m.reshape(size * size)
            m = m[m != -np.inf]
            mean_value = np.mean(m)
            all_mean.append(mean_value)
            
            f.write("euclidean_distances -" + str(categories[i]) + ".: " + str(mean_value) + '\n')
            
        # overall
        mean_overall = np.sum(all_mean) / len(np.unique(y))
        f.write("euclidean_distances - overall.: " + str(mean_overall) + '\n')    
        f.close()
    
    def pearson_correlation_index(self, args, X, y, categories):
        
        file_name = os.path.join(args.savedir, 'tsne_epoch_'+ str(self.trainer.current_epoch) + '_pearson_correlation.txt')
        f = open(file_name, 'w')
        f.write("--- Embedding evaluation --- \n")
        
        all_mean = [] 
        for i in np.unique(y):
            index_class = np.where(y==i)
            m = np.corrcoef(X[index_class])
            # calculating the off-diagonal elements
            np.fill_diagonal(m, -np.inf)
            size = len(m)
            m = m.reshape(size * size)
            m = m[m != -np.inf]
            mean_value = np.mean(m)
            all_mean.append(mean_value)
            
            f.write("pearson_correlation -" + str(categories[i]) + ".: " + str(mean_value) + '\n')
            
        # overall
        mean_overall = np.sum(all_mean) / len(np.unique(y))
        f.write("pearson_correlation - overall.: " + str(mean_overall) + '\n')    
        f.close()
    
    def spearman_correlation_index(self, args, X, y, categories):
        
        file_name = os.path.join(args.savedir, 'tsne_epoch_'+ str(self.trainer.current_epoch) + '_spearman_correlation.txt')
        f = open(file_name, 'w')
        f.write("--- Embedding evaluation --- \n")
        
        all_mean = [] 
        for i in np.unique(y):
            index_class = np.where(y==i)
            res = spearmanr(X[index_class], axis=1)
            m = res.correlation
            # calculating the off-diagonal elements
            np.fill_diagonal(m, -np.inf)
            size = len(m)
            m = m.reshape(size * size)
            m = m[m != -np.inf]
            mean_value = np.mean(m)
            all_mean.append(mean_value)
            
            f.write("spearman_correlation -" + str(categories[i]) + ".: " + str(mean_value) + '\n')
            
        # overall
        mean_overall = np.sum(all_mean) / len(np.unique(y))
        f.write("spearman_correlation - overall.: " + str(mean_overall) + '\n')    
        f.close()
    
    def plot_embeddings3D(self, args, X, y, categories):

        X = X.permute(0, 2, 3, 1).contiguous()
        shape = X.size()
        X = X.view(shape[0] * shape[1] * shape[2], shape[3])
        X = X.detach().cpu().numpy()
        
        y = y.view(shape[0] * shape[1] * shape[2])
        y = y.detach().cpu().numpy()

        # print(X.shape)

        # remove ignore index
        mask = np.ones(len(y), dtype=bool)
        mask[y == 255] = False
        X = X[mask,...]
        y = y[mask,...]
        
        # random get indx
        # indices = np.random.randint(X.shape[0], size=10000)
        # X = X[indices]
        # y = y[indices]
        X, y = self.select_n_emb_per_class(X, y, 5000)
        
        # get axes embeddings
        X = X.T
        Xax = X[0]
        Yax = X[1]
        Zax = X[2]
        
        # dict
        cdict = {0:'#333333', 1:'#0343ff', 2:'#15b01a'}
        labl = dict(zip(range(3), categories))
        # print(labl)
        marker = {0:'*',1:'s',2:'o'}
        alpha = {0:.3, 1:.5, 2:.7}
        
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111, projection="3d")
        # ax.force_zorder = True
        
        fig.patch.set_facecolor('white')
        for l in np.unique(y):
            ix=np.where(y==l)
            # ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
            #     label=labl[l], marker=marker[l], alpha=alpha[l])
            ax.plot(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], markersize=3,
                label=labl[l], marker=marker[l], alpha=alpha[l], linewidth=0)
        
        # # make simple, plot anchors
        if args.loss == 'prototycal_triplet' or args.loss == 'prototycal_random':
            ax.plot(3, 0, 0, c='r', marker=marker[0], markersize=8)
            ax.plot(0, 3, 0, c='r', marker=marker[1], markersize=8)
            ax.plot(0, 0, 3, c='r', marker=marker[2], markersize=8)
        
        # for loop ends
        # ax.set_xlabel("First Principal Component", fontsize=14)
        # ax.set_ylabel("Second Principal Component", fontsize=14)
        # ax.set_zlabel("Third Principal Component", fontsize=14)
        legend = ax.legend(loc='best', markerscale=4, fontsize=14)
        
        plt.gca().view_init(20, 45)
        # plt.show()
        
        file_path = os.path.join(args.savedir,'plot_epoch_'+ str(self.trainer.current_epoch)+'.png')
        plt.savefig(file_path)
        
    def plot_batch_embeddings3D(self, args, X, y, categories, batch_idx):

        X = X.permute(0, 2, 3, 1).contiguous()
        shape = X.size()
        X = X.view(shape[0] * shape[1] * shape[2], shape[3])
        X = X.detach().cpu().numpy()
        
        y = y.view(shape[0] * shape[1] * shape[2])
        y = y.detach().cpu().numpy()

        # print(X.shape)

        # remove ignore index
        mask = np.ones(len(y), dtype=bool)
        mask[y == 255] = False
        X = X[mask,...]
        y = y[mask,...]
        
        # random get indx
        # indices = np.random.randint(X.shape[0], size=10000)
        # X = X[indices]
        # y = y[indices]
        X, y = self.select_n_emb_per_class(X, y, 5000)
        
        # get axes embeddings
        X = X.T
        Xax = X[0]
        Yax = X[1]
        Zax = X[2]
        
        # dict
        cdict = {0:'#333333', 1:'#0343ff', 2:'#15b01a'}
        labl = dict(zip(range(3), categories))
        # print(labl)
        marker = {0:'*',1:'s',2:'o'}
        alpha = {0:.3, 1:.5, 2:.7}
        
        fig = plt.figure(figsize=(7,5))
        ax = fig.add_subplot(111, projection="3d")
        # ax.force_zorder = True
        
        fig.patch.set_facecolor('white')
        for l in np.unique(y):
            ix=np.where(y==l)
            # ax.scatter(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], s=40,
            #     label=labl[l], marker=marker[l], alpha=alpha[l])
            ax.plot(Xax[ix], Yax[ix], Zax[ix], c=cdict[l], markersize=3,
                label=labl[l], marker=marker[l], alpha=alpha[l], linewidth=0)
        
        # # make simple, plot anchors
        if args.loss == 'prototycal_triplet' or args.loss == 'prototycal_random':
            ax.plot(3, 0, 0, c='r', marker=marker[0], markersize=8)
            ax.plot(0, 3, 0, c='r', marker=marker[1], markersize=8)
            ax.plot(0, 0, 3, c='r', marker=marker[2], markersize=8)
        
        #adding text inside the plot  
        plt.title('Epoch.: ' + str(self.trainer.current_epoch) + ' > Batch.: ' + str(batch_idx), loc='left', fontdict={'fontsize': 18})    
        
        # for loop ends
        # ax.set_xlabel("First Principal Component", fontsize=14)
        # ax.set_ylabel("Second Principal Component", fontsize=14)
        # ax.set_zlabel("Third Principal Component", fontsize=14)
        legend = ax.legend(loc='best', markerscale=4, fontsize=14)
        
        plt.gca().view_init(20, 45)
        # plt.show()
        
        file_path = os.path.join(args.savedir, 'batch', 'plot_epoch_'+ str(self.trainer.current_epoch)+'_batch_' + str(batch_idx) + '.png')
        plt.savefig(file_path)


