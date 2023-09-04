import torch
import models
import train
from torch_geometric.datasets import MoleculeNet
from torch_geometric import nn
import warnings

warnings.filterwarnings("ignore")


def run():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the ESOL dataset
    data = MoleculeNet(root=".", name="ESOL")

    # Load the model
    model = models.GCN(arch=nn.GCNConv,
                       num_features=data.num_features,
                       embedding_size=64)

    # Root mean squared error and optimzer
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)

    # get data loaders
    loader, test_loader = train.get_dataloaders(data, split=0.8, batch_size=64)

    # train the model
    losses = train.train(optimizer, loss_fn, model, loader, device)

    # save the model
    torch.save(model.state_dict(), '../trained_models/GCNConv_1.pth')

    #evaluate model
    evaluate_test(model, test_loader, device, loss_fn)


def evaluate_test(model, test_loader, device, loss_fn):
    predictions = []
    actuals = []
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch.to(device)
            pred, embed = model(test_batch.x.float(), test_batch.edge_index,
                                test_batch.batch)
            #loss = loss_fn(pred, test_batch.y)
            predictions += pred.tolist()
            actuals += test_batch.y.tolist()

    loss = loss_fn(torch.Tensor(predictions), torch.Tensor(actuals))
    print(f"Test Loss = {loss}")


if __name__ == "__main__":
    run()
