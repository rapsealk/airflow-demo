import torch
import torch.nn.functional as F
import torch.optim as optim


class Trainer:
    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    def train(
        self,
        args,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
    ) -> None:
        self._model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self._model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                ))
                if args.dry_run:
                    break

    def test(
        self,
        device: torch.device,
        test_loader: torch.utils.data.DataLoader,
    ) -> None:
        self._model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self._model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
        ))
