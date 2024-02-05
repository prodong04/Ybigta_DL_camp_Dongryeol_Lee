import argparse
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms as T
from tqdm.auto import tqdm

from dataset import FoodDataset
from model import vanillaCNN, vanillaCNN2, VGG19




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=['CNN1', 'CNN2', 'VGG'], required=True, help='model architecture to train')
    parser.add_argument('-e', '--epoch', type=int, default=100, help='the number of train epochs')
    parser.add_argument('-b', '--batch', type=int, default=32, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='learning rate')
    return parser.parse_args()



if __name__ == '__main__':
    args = parse_args()
    
    os.makedirs('./save', exist_ok=True)
    os.makedirs(f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}', exist_ok=True)
    
    transforms = T.Compose([
        T.Resize((227,227), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomVerticalFlip(0.5),
        T.RandomHorizontalFlip(0.5),
    ])

    train_dataset = FoodDataset("./data", "train", transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataset = FoodDataset("./data", "val", transforms=transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    if args.model == 'CNN1':
        model = vanillaCNN()
    elif args.model == 'CNN2':
        model = vanillaCNN2()
    elif args.model == 'VGG': 
        model = VGG19()
    else:
        raise ValueError("model not supported")
        
    ##########################   fill here   ###########################
        
    # TODO : Training Loop을 작성해주세요
    # 1. logger, optimizer, criterion(loss function)을 정의합니다.
    # train loader는 training에 val loader는 epoch 성능 측정에 사용됩니다.
    # torch.save()를 이용해 epoch마다 model이 저장되도록 해 주세요
    ######################################################################

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    model.to(device)
    criterion.to(device)

    log_path = f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}/log.txt'
    log_format = '%(asctime)s:%(levelname)s:%(message)s'
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG, format=log_format)
    logger = logging.getLogger(__name__)
    console_handler = logging.StreamHandler()  # 콘솔 핸들러 생성
    console_handler.setLevel(logging.DEBUG)      # 원하는 로그 레벨 설정
    console_handler.setFormatter(logging.Formatter(log_format))  # 로그 포맷 설정
    logger.addHandler(console_handler)
    lst = []
    x = []
    y = []
    for epoch in range(args.epoch):
        logger.info(f'Training epoch {epoch}')
        model.train()
        train_loss = 0
        #그래프를 위해 리스트 생성
        lst.append(epoch+1)
        train_acc = 0
        for i, (x, y) in enumerate(tqdm(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (pred.argmax(1) == y).sum().item()
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        logger.info(f'epoch {epoch} = Train Score {train_acc*100:.4f}%)')
        

        model.eval()
        val_loss = 0
        val_acc = 0
        logger.info(f'Validating epoch {epoch}')
        with torch.no_grad():
            for i, (x, y) in enumerate(tqdm(val_loader)):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                y.append(loss.item())
                val_acc += (pred.argmax(1) == y).sum().item()
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        logger.info(f'epoch {epoch} = Val score {val_acc * 100:.4f}%')
        
        torch.save(model.state_dict(), f'./save/{args.model}_{args.epoch}_{args.batch}_{args.learning_rate}/epoch_{epoch}.pt')
        x.append(train_loss)
        y.append(val_loss)
    print(f'Epoch {epoch} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}')
    # Separate plots for train_loss and val_loss
    x_cpu = x.cpu().numpy
    y_cpu = y.cpu().numpy
    plt.subplot(2, 1, 1)
    plt.plot(lst, x_cpu, label='train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(lst, y_cpu, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

