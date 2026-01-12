import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import os
import sys

device = torch.device('cpu')
print(f"Rulez pe: {device}")

if not os.path.exists('ACM.mat'):
    print("EROARE: Lipseste fisierul ACM.mat")
    sys.exit()

try:
    data_mat = loadmat('ACM.mat')
    A_raw = data_mat['Network']
    X_raw = data_mat['Attributes']
    
    if hasattr(X_raw, 'toarray'):
        X_np = X_raw.toarray().astype(np.float32)
    else:
        X_np = X_raw.astype(np.float32)
    X = torch.tensor(X_np)
    del X_np 

    A_coo = coo_matrix(A_raw)
    row = torch.from_numpy(A_coo.row.astype(np.int64))
    col = torch.from_numpy(A_coo.col.astype(np.int64))
    edge_index = torch.stack([row, col], dim=0)

    if hasattr(A_raw, 'toarray'):
        A_true = torch.tensor(A_raw.toarray().astype(np.float32))
    else:
        A_true = torch.tensor(A_raw.astype(np.float32))

except Exception as e:
    print(f"Eroare date: {e}")
    sys.exit()

X = X.to(device)
A_true = A_true.to(device)
edge_index = edge_index.to(device)

class GraphAutoencoder(torch.nn.Module):
    def __init__(self, input_dim):
        super(GraphAutoencoder, self).__init__()
        self.enc1 = GCNConv(input_dim, 16) 
        self.enc2 = GCNConv(16, 8)
        
        self.attr_dec = GCNConv(8, input_dim)
        self.struct_dec = GCNConv(8, 8)

    def forward(self, x, edge_index):
        z = self.enc1(x, edge_index)
        z = F.relu(z)
        z = self.enc2(z, edge_index)
        z = F.relu(z)
        
        x_hat = self.attr_dec(z, edge_index)
        
        z_struct = self.struct_dec(z, edge_index)
        z_struct = F.relu(z_struct)
        a_hat = torch.mm(z_struct, z_struct.t())
        
        return x_hat, a_hat

model = GraphAutoencoder(input_dim=X.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

alpha = 0.5
epochs = 15
losses = []

print("Start Antrenare...")

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    x_hat, a_hat = model(X, edge_index)
    
    loss_attr = torch.mean((X - x_hat)**2)
    loss_struct = torch.mean((A_true - a_hat)**2)
    
    loss = alpha * loss_attr + (1 - alpha) * loss_struct
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    print(f'Epoca {epoch+1}/{epochs} | Loss: {loss.item():.4f}')

plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Loss')
plt.show()

model.eval()
with torch.no_grad():
    x_hat, _ = model(X, edge_index)
    errors = torch.mean((X - x_hat)**2, dim=1).cpu().numpy()

plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, color='green')
plt.title('Erori Reconstructie')
plt.show()