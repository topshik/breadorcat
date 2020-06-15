import datetime
from typing import Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from tqdm import tqdm


class BreadOr(torch.nn.Module):
    def __init__(self, vgg: torch.nn.Module) -> None:
        super().__init__()
        features = []
        for i, module in enumerate(vgg.features):
            if i <= 23:
                features.append(module)
        self.features = torch.nn.Sequential(*features)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = torch.nn.Sequential(torch.nn.Linear(7 * 7 * 512, 2048),
                                              torch.nn.ReLU(inplace=True),
                                              torch.nn.Dropout(),
                                              torch.nn.Linear(2048, 1024),
                                              torch.nn.ReLU(inplace=True),
                                              torch.nn.Dropout(),
                                              torch.nn.Linear(1024, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.avg_pool(feats)
        fltnd = torch.nn.Flatten()(pooled)
        return self.classifier(fltnd)


# noinspection PyUnresolvedReferences
def train(model: torch.nn.Module, n_epochs: int = 100, batch_size: int = 10, n_workers: int = 4,
          checkpoint_path: str = './breador.pth') -> Tuple[float, float]:
    # set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: GPU")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # load data
    transform = {
        'train': transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05),
                transforms.ToTensor(),
            ]),
        'test': transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
    }
    train_dataset = torchvision.datasets.ImageFolder('data/train', transform=transform['train'])
    test_dataset = torchvision.datasets.ImageFolder('data/test', transform=transform['test'])
    print(f'{len(train_dataset)} training images')
    print(f'{len(test_dataset)} test images')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=n_workers,
                              shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=n_workers,
                             shuffle=False, pin_memory=True)

    print(f'Train/test data loaders have {len(train_loader)} and {len(test_loader)} batches')

    # init net, optimizer and loss
    vgg16 = torchvision.models.vgg16()
    breador = BreadOr(vgg16)
    breador.to(device)
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(breador.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    date = datetime.datetime.now().strftime('%b-%d-%Y-%H:%M:%S')
    writer_train = SummaryWriter(f'runs/{date}/train')
    writer_test = SummaryWriter(f'runs/{date}/test')
    best_acc, train_acc, test_acc = 0, 0, 0

    for i in range(n_epochs):
        model.train()
        correct, total = 0, 0
        for j, (images, labels) in enumerate(tqdm(train_loader)):
            probs = model(images.to(device))
            with torch.no_grad():
                labels = labels.to(device)
                predictions = probs.max(1)[1]

                total += len(labels)
                correct += (predictions == labels).sum().item()

            loss = criterion(probs, labels)
            writer_train.add_scalar('Loss', loss, i * len(train_loader) + j)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = correct / total
        writer_train.add_scalar('Accuracy', train_acc, i)

        model.eval()
        correct, total = 0, 0
        for j, (images, labels) in enumerate(tqdm(test_loader)):
            probs = model(images.to(device))
            labels = labels.to(device)
            predictions = probs.max(1)[1]
            total += len(labels)
            correct += (predictions == labels).sum().item()
            val_loss = criterion(probs, labels)
            writer_test.add_scalar('Loss', val_loss,
                                   (i * len(test_loader) + j) * len(train_loader) / len(test_loader))
        test_acc = correct / total
        writer_test.add_scalar('Accuracy', test_acc, i)
        print(f'Epoch number: {i}')
        print(f'Train accuracy: {train_acc}')
        print(f'Test accuracy: {test_acc}')
        if test_acc > best_acc:
            torch.save(model.state_dict(), checkpoint_path)
            best_acc = test_acc

    return train_acc, test_acc


def evaluate(model_path: str = 'breador.pth') -> None:
    # set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: GPU')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor()])
    test_dataset = torchvision.datasets.ImageFolder('data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10, num_workers=4, shuffle=False, pin_memory=True)

    vgg16 = torchvision.models.vgg16()
    breador_loaded = BreadOr(vgg16)
    breador_loaded.load_state_dict(torch.load(model_path, map_location='cpu'))
    breador_loaded.to(device)
    breador_loaded.eval()
    correct, total = 0, 0

    for j, (images, labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            probs = breador_loaded(images.to(device))
            labels = labels.to(device)
            predictions = probs.max(1)[1]
            total += len(labels)
            correct += (predictions == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f'Validation accuracy: {val_accuracy:.2f}%')
