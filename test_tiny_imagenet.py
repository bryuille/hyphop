import argparse
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from datasets.tiny_imagenet import get_tiny_imagenet_loaders
from models.wrappers import SingleInstanceClassifier as ModelWrapper
from utils.device import select_device


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Core Tiny ImageNet')
    parser.add_argument('--model', type=str, default='kf_attention', 
                        choices=['kf_attention', 'kf_layer', 'kf_pooling', 
                                 'hf_attention', 'hf_layer', 'hf_pooling',
                                 'ein_attention', 'ein_layer', 'ein_pooling'])
    parser.add_argument('--data-dir', type=str, default='./datasets',
                        help='root directory for Tiny ImageNet (default: ./datasets)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.96, metavar='M',
                        help='Learning rate step gamma (default: 0.96)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='hidden dimension d (default: 128)')
    parser.add_argument('--beta', type=float, default=None,
                        help='scaling factor beta; default derives from d')
    parser.add_argument('--num-states', type=int, default=1,
                        help='num pooling states (default: 1)')
    parser.add_argument('--num-memories', type=int, default=64,
                        help='num static memories (default: 64)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers (default: 2)')
    parser.add_argument('--no-accel', action='store_true',
                        help='disables accelerator')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='device to use (default: auto)')
    parser.add_argument('--dry-run', action='store_true',
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', 
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = select_device(args.device, no_accel=args.no_accel)

    train_loader, test_loader, input_dim, num_classes = get_tiny_imagenet_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type != "cpu"),
    )

    beta = args.beta if args.beta is not None else 1 / math.sqrt(args.hidden_dim)
    model = ModelWrapper(
        mode=args.model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        beta=beta,
        num_states=args.num_states,
        num_memories=args.num_memories,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "tiny_imagenet_core.pt")

    final_acc = test(model, device, test_loader)
    return final_acc


if __name__ == "__main__":
    main()

