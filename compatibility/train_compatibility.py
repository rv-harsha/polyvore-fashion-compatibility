import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
import time
import copy
from tqdm import tqdm
import os.path as osp

from utils import Config
from model import model
from data import get_dataloader
from datetime import datetime


def train_model(dataloader, model, criterion, optimizer, device, num_epochs, dataset_size):

    ts = datetime.now().strftime('%Y%m%d_%H%M%S');
    writer = SummaryWriter(f'logs/polyvore/{ts}');

    model.to(device)
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'test']:
            print('   Phase: {}'.format(phase))
            print('   ','-' * 10)

            if phase=='train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, pred = outputs.max(1)
                    total += labels.size(0)
                    loss = criterion(outputs, labels)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += pred.eq(labels).sum().item()

            epoch_loss = running_loss / dataset_size[phase]
            epoch_acc = running_corrects * 100 / total

            writer.add_scalars('data/loss', {phase: epoch_acc}, epoch+1)
            writer.add_scalars('data/accuracy', {phase: epoch_loss}, epoch+1)

            print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

            if phase=='test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        torch.save(best_model_wts, osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth'))
        print('Model saved at: {}'.format(osp.join(Config['root_path'], Config['checkpoint_path'], 'model.pth')))

    time_elapsed = time.time() - since
    print('Time taken to complete training: {:0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best acc: {:.4f}'.format(best_acc))
    writer.close()

    # model.eval()
    # corrects = 0
    # test_loader = dataloaders['test']
   
    # with open(osp.join(Config['root_path'], "category.csv"), 'w') as csv_file:
    #     writer = csv.writer(csv_file)
    #     with torch.no_grad():
    #         for i, (inputs, labels, item_ids) in tqdm(enumerate(test_loader), 0):

    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             outputs = model(inputs)
    #             _, pred = outputs.max(1)

    #             total += labels.size(0)
    #             corrects += pred.eq(labels).sum().item()

    #             preds_list = pred.cpu().detach().numpy().tolist()
    #             labels_list = labels.cpu().detach().numpy().tolist()

    #             stacked_data = np.column_stack((list(item_ids), preds_list, labels_list))
    #             writer.writerows(stacked_data)

    #         final_acc = corrects * 100 / total
    #         print('Best Test Accuracy: {:.4f}'.format(final_acc))

if __name__=='__main__':

    dataloaders, classes, dataset_size = get_dataloader(debug=Config['debug'], batch_size=Config['batch_size'], num_workers=Config['num_workers'])
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=Config['learning_rate'])
    device = torch.device('cuda:0' if torch.cuda.is_available() and Config['use_cuda'] else 'cpu')
    train_model(dataloaders, model, criterion, optimizer, device, num_epochs=Config['num_epochs'], dataset_size=dataset_size)