import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

def corrdist(x,y):
    k = x.shape[0]
    y = y.view(y.shape[0],1,y.shape[1])
    A = torch.concat([x,y])
    c = torch.corrcoef(A.squeeze())
    return 1.-c[0,k:]

def plus_plus(dataset, k, random_state=42, batch_size=32, device="cpu", p=2., dist='p'):
    """
    Create cluster centroids using the k-means++ algorithm.
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    """

    torch.manual_seed(random_state)
    centroids = torch.stack([dataset[0][0]], 0)
    centroids = centroids.view(1,1,centroids.shape[-1]).to(device)
    loader = DataLoader(dataset, batch_size=batch_size)
    print("Initializing KMeans")
    for ik in range(1, k):
        print("Initializing cluster %d" % ik)      
        #r = torch.rand().to(device)
        max_dist = 0.
        candidate = None
        pbar = tqdm.tqdm(enumerate(loader),total=len(loader))
        for i, batch in pbar:  
            samples, _ = batch
            samples = samples.to(device)
            if "corr" in dist.lower():
                dist_sq = torch.pow(corrdist(centroids, samples), 2.)
            else:
                dist_sq = torch.pow(torch.cdist(centroids, samples, p=p),p)
            dist_sq,_ = dist_sq.min(0)
            dist_sq = dist_sq.squeeze()
            dist_sq = dist_sq/dist_sq.sum()
            indices = dist_sq.argmax()
            dist_max = dist_sq[indices]
            new_candidate = samples[indices]
            if dist_max > max_dist:
                max_dist = dist_max
                candidate = new_candidate#.clone()
            pbar.set_description("Max Distance %.5f; Batch %.5f" % (max_dist, dist_max))
            #pbar.update(1)
        centroids = torch.concat([centroids, candidate.view(1,1,candidate.shape[-1])])            
    return centroids

class Kmeans(nn.Module):
    def __init__(self, 
                 num_clusters, 
                 init_method="kmeans++",
                 dist='p',
                 p=2.):
        super().__init__()
        self.num_clusters = num_clusters
        self.init_method = init_method
        self.centroids = None
        self.dist = dist
        self.p = p

    def initialize(self, dataset, random_state=42, batch_size=32, device="cpu"):
        self.centroids = plus_plus(dataset, self.num_clusters, random_state, batch_size, device, self.p, self.dist)

    def forward(self, x):
        if "corr" in self.dist.lower():
            dist_sq = torch.pow(corrdist(self.centroids, x), 2.)
        else:
            dist_sq = torch.pow(torch.cdist(self.centroids, x, p=self.p),self.p)
        assignment = dist_sq.argmin(0).squeeze()
        new_means = torch.zeros_like(self.centroids, device=self.centroids.device)
        for k in range(self.num_clusters):
            new_means[k, ...] = x[torch.where(assignment==k)].mean(0)
        return assignment, new_means
        

if __name__=="__main__":
    from torchvision.datasets import MNIST
    from torchvision import transforms
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))
        ])
    batch_size = 1024
    dataset = MNIST(root="./data/",download=True, transform=transform)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    km = Kmeans(10)
    km.initialize(dataset,batch_size=batch_size, device=device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=12, prefetch_factor=2)
    
    #km
    movement = 99999
    threshold = 1e-9
    while movement > threshold:
        pbar = tqdm.tqdm(enumerate(loader),total=len(loader))
        initial_clusters = km.centroids.clone()
        new_clusters = initial_clusters.clone()
        for i, batch in pbar:
            sample, _ = batch
            sample = sample.to(device)
            assignment, clusters = km(sample)
            new_clusters += clusters
            new_clusters /= 2.            
            movement =  torch.pow(torch.cdist(km.centroids, new_clusters, p=2),2).sum()
            pbar.set_description("Movement %.4f" % (movement.item()))
        movement =  torch.pow(torch.cdist(km.centroids, new_clusters, p=2),2).sum()
        pbar.set_description("Movement %.4f" % (movement.item()))
        km.centroids = new_clusters.clone()
        if movement < threshold:
            break
    #result = plus_plus(dataset, 5, device='cuda' if torch.cuda.is_available() else "cpu")
