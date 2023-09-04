import torch
from torch_geometric.data import DataLoader
import warnings

warnings.filterwarnings("ignore")


def get_dataloaders(data, split, batch_size):
    # Root mean squared error
    # loss_fn = torch.nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

    # Wrap data in a data loader
    data_size = len(data)
    #NUM_GRAPHS_PER_BATCH = 64
    loader = DataLoader(data[:int(data_size * split)],
                        batch_size=batch_size,
                        shuffle=True)
    test_loader = DataLoader(data[int(data_size * split):],
                             batch_size=batch_size,
                             shuffle=True)

    return loader, test_loader


def train_epoch(loader, device, optimizer, loss_fn, model):
    # Enumerate over the data
    for batch in loader:
        # Use GPU
        batch.to(device)
        # Reset gradients
        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch)
        # Calculating the loss and gradients
        loss = loss_fn(pred, batch.y)
        loss.backward()
        # Update using the gradients
        optimizer.step()
    return loss, embedding


def train(optimizer, loss_fn, model, loader, device):
    # Use GPU for training
    model = model.to(device)

    print("Starting training...")
    losses = []
    for epoch in range(2000):
        loss, h = train_epoch(loader, device, optimizer, loss_fn, model)
        losses.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Train Loss {loss}")

    return losses
