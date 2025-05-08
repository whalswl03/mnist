import torch
import torchvision
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from dataset import get_dataloaders
from train import train
from test import test
from model import Net

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_accel and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    train_loader, val_loader, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=1)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=10, gamma=args.gamma)
   
    train_losses = []
    val_accuracies = []

    for epoch in range(1, args.epochs + 1):
        loss = train(args, model, device, train_loader, optimizer, epoch)
        val_accuracy = test(model, device, val_loader)
        train_losses.append(loss)
        val_accuracies.append(val_accuracy)
        scheduler.step()
    
    print("===Test accuracy===")
    test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    torch.save({
        "losses": train_losses,
        "accuracies": val_accuracies
    }, "logs.pt")


if __name__ == '__main__':
    main()