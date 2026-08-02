# Don't erase the template code, except "Your code here" comments.

import subprocess
import sys
import os

# List any extra packages you need here. Please, fix versions so reproduction of your results would be less painful.
PACKAGES_TO_INSTALL = ["gdown==4.4.0", "timm"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + PACKAGES_TO_INSTALL)

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch import nn

import numpy as np
from tqdm import tqdm
import wandb
import timm 


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
loss_fn = nn.CrossEntropyLoss()
print(device)

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super(CNNBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool= nn.MaxPool2d((2, 2)) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

def get_dataloader(path, kind, transform: transforms.ToTensor | None = None, batch_size: int | None = None):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val' or 'test', the dataloader should be deterministic.
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train', 'val' or 'test'
    transform:
        `torchvision.transforms` object
        If `None`, defaults to `transforms.ToTensor()`.

    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    if transform is None:
        transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.RandomRotation(30),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ]
)
    if batch_size is None:
        batch_size = 100
    
    drop_last = shuffle = kind == 'train'
    if drop_last:
        dataset = datasets.ImageFolder(path+kind, transform=transform)
    else:
        dataset = datasets.ImageFolder(path+kind, transform=transforms.Compose(
    [   transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            ),
    ]))
    num_workers = 4
    return DataLoader(dataset, 
                      batch_size=batch_size,
                      shuffle=shuffle,
                      drop_last=drop_last,
                      num_workers=num_workers,
                      pin_memory=True,
                      prefetch_factor=4,)
    


def get_model(name: str | None = None):
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.

    return:
    model:
        `torch.nn.Module`
    """
    if name is None:
        model = nn.Sequential(
        CNNBlock(3, 64),
        CNNBlock(64, 128),
        CNNBlock(128, 256),
        CNNBlock(256, 512),
        CNNBlock(512, 1024),
        nn.AdaptiveAvgPool2d((1, 1)),  # B x 1024 x 1 x 1
        nn.Flatten(),  # B x 1024
        nn.Dropout(0.5),
        nn.Linear(1024, 200),
        )
    else:
        model = timm.create_model(name, pretrained=False, num_classes=200)
    model.apply(init_weights)

    return model.to(device)

def get_optimizer(model, 
                  optimizer_algo:torch.optim.Optimizer | None = None, 
                  learning_rate: float | None = None,
                  weight_decay: float | None = None):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.

    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    if learning_rate is None:
        learning_rate = 3e-4
    if weight_decay is None:
        weight_decay = 0.05
    torch.manual_seed(111)
    torch.cuda.manual_seed(111)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if optimizer_algo is None:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer_algo(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).

    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """

    logits = model(batch.to(device)).cpu()
    return logits

def validate(dataloader, model, epoch: int| None = None, loss_fn: nn.Module | None = None):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.

    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    if epoch is None:
        epoch = "Not given"
        
    return run_epoch("val", model, dataloader, loss_fn, None, epoch, device)

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer, 
                          num_epochs:int | None = None, checkpoint_path: str | None = None,
                          start_epoch=0):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.

    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    if num_epochs is None:
        num_epochs = 5
    if checkpoint_path is None:
        checkpoint_path = "checkpoints_baseline"

    
    
    train_losses = []
    val_losses = []
    
    train_accs = []
    val_accs = []

    best_val_acc = -np.inf
    best_val_acc_epoch = -1
    best_val_acc_fn = None

    os.makedirs(checkpoint_path, exist_ok=True)
    
    scheduler  = LRScheduler(optimizer, patience=0, factor=0.2)
    
    for epoch in range(num_epochs):
        torch.manual_seed(epoch)
        torch.cuda.manual_seed(epoch)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
            
        train_acc, train_loss  = run_epoch("train", model, train_dataloader, loss_fn, optimizer, epoch, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        val_acc, val_loss  = validate(dataloader=val_dataloader, model=model, loss_fn=loss_fn, epoch=epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        wandb.log({"epoch_loss_train": train_loss, "epoch_loss_val": val_loss, 
                   "epoch_accuracy_train": train_acc, "epoch_accuracy_val": val_acc, 
                   "epoch": start_epoch + epoch})
        scheduler(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch

            output_fn = os.path.join(checkpoint_path, f"epoch={str(epoch).zfill(2)}_valacc={best_val_acc:.3f}.pth")
            save_weights(model, output_fn)
            print(f"New checkpoint saved to {checkpoint_path}\n best_val_accuracy {best_val_acc}")
            
            best_val_acc_fn = output_fn
            
        torch.cuda.empty_cache()


    print(f"Best val_acc = {best_val_acc:.3f} reached at epoch {best_val_acc_epoch}")
    load_weights(model, best_val_acc_fn)
    save_weights(model, "./checkpoint.pth")
    wandb.finish()
    return train_losses, val_losses, train_accs, val_accs, best_val_acc, model

def save_weights(model, checkpoint_path):
    with open(checkpoint_path, "wb") as fp:
        torch.save(model.state_dict(), fp)
        
def load_weights(model, output_fn):
    """
    Initialize `model`'s weights from `checkpoint_path` file.

    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    with open(output_fn, "rb") as fp:
        state_dict = torch.load(fp, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    

def run_epoch(stage, model, dataloader, loss_fn, optimizer, epoch, device):
    if stage == "train":
        model.train()
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)
        model.eval()

    model = model.to(device)

    losses = []
    accs = []
    with tqdm(total=len(dataloader), desc=f"epoch: {str(epoch).zfill(3)} | {stage:5}", ncols=80) as pbar:
        for batch in dataloader:
            xs, ys_true = batch
            ys_pred = model(xs.to(device))
            loss = loss_fn(ys_pred, ys_true.to(device))
            acc = get_accuracy(ys_pred, ys_true)

            if stage == "train":
                loss.backward()
                # for i in range(5):
                #     model[i].conv1.weight.grad += torch.randn_like(model[i].conv1.weight.grad)/10
                #     model[i].conv2.weight.grad += torch.randn_like(model[i].conv2.weight.grad)/10
                    
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                optimizer.zero_grad()
                wandb.log({"lr": optimizer.param_groups[0]["lr"], "loss": loss})
            losses.append(loss.detach().cpu().item())
            accs.append(acc.detach().cpu().item())
            pbar.update(1)

    return np.mean(accs), np.mean(losses)

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    
    md5_checksum = "cfe1c3a7dc04f223e6ff5f0d666d2f3c"
    google_drive_link = "https://drive.google.com/file/d/1epcrBSjpTkbNRYpDWLkcrw4avfc5zdfR/"
    return md5_checksum, google_drive_link

def get_accuracy(model_probs, labels):
    _, max_prob_index = torch.max(model_probs, dim=1)
    
    return torch.mean((max_prob_index.detach().cpu() == labels).float())



class LRScheduler():
    """
    Learning rate scheduler. If the validation acc does not increase for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.1
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='max',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
