#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from tqdm import tqdm
from CNN import CNN, CNN_deep
from MLP import MLP, MLP_shallow, MLP_deep

def print_title(title):
    print('-'*80)
    print('| ', title)
    print('-'*80)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {DEVICE}')

DS = datasets.CIFAR10 # MNIST, CIFAR10

N_BATCH = 32 # mini-batch size
MODEL_NAME = 'CNN_deep' # MLP, CNN, CNN_deep
print(f'Model: {MODEL_NAME}')
# LR = 1e-1
# LR = 1e-2
LR = 1e-3
EPOCH = 5

TRAIN_MODEL = False               # 새로 학습할지 여부
SAVE_MODEL = True and TRAIN_MODEL # 학습한 경우, 모델을 파일에 저장할지 여부
LOAD_MODEL = not TRAIN_MODEL      # 테스트를 위해 저장된 모델을 사용할지 여부(True: 저장된 모델, False: 학습한 모델)

DATA_PATH = '../data' # where to save data
MODEL_PATH = '../result' # where to save model

print_title('Data Section')
# Dataset
DS_NAME = DS.__name__
print(f'Dataset: {DS_NAME}')

transform = transforms.ToTensor()
train_DS = DS(root=DATA_PATH, train=True, download=True, transform=transform)
test_DS = DS(root=DATA_PATH, train=False, download=True, transform=transform)
print(train_DS)
print(test_DS)

print(f'Number of classes : {len(train_DS.classes)}')
print(f'Classes : {train_DS.classes}')
print(f'Class-to-index map : {train_DS.class_to_idx}')
plt.figure(figsize=(8, 6))
for i in range(12):
  plt.subplot(3, 4, i+1)
  # plt.subplot(3, 4, i+1, xticks=[], yticks=[])
  plt.imshow(train_DS.data[i], cmap='gray')
  plt.title(train_DS.classes[train_DS.targets[i]], color='k')
  plt.axis('off')

# Dataloader
train_DL = torch.utils.data.DataLoader(train_DS, batch_size=N_BATCH, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=N_BATCH, shuffle=False)

N_TRAIN = len(train_DS)
N_TEST = len(test_DS)
print(f'Number of training data : {N_TRAIN}')
print(f'Number of test data : {N_TEST}')

N_CLASSES = len(train_DS.classes)
print(f'Number of classes: {N_CLASSES}')

x_batch, y_batch = next(iter(train_DL))
N_CH = x_batch.shape[1]
N_WIDTH = x_batch.shape[2]
N_HEIGHT = x_batch.shape[3]
print(f'Channel: {N_CH}, Width: {N_WIDTH}, Height: {N_HEIGHT}')

# Model
model = globals()[MODEL_NAME](N_CH, N_WIDTH, N_HEIGHT, N_CLASSES).to(DEVICE)
print_title('Model Section')
print(model)
x_batch, y_batch = next(iter(train_DL))
x_batch = x_batch.to(DEVICE)
y_batch = y_batch.to(DEVICE)
print(x_batch.shape)
print(y_batch.shape)
y = model(x_batch.to(DEVICE))
print(y.shape, y[0])
print(y_batch.shape, y_batch[0])


# Parameter 수 구하기
def count_params(model):
  return sum([p.numel() for p in model.parameters()])
print(f'Number of parameters : {count_params(model)}')


# Train
if TRAIN_MODEL:
  print_title('Training Section')
  # optimizer = optim.SGD(model.parameters(), lr=LR)
  optimizer = optim.Adam(model.parameters(), lr=LR)

  criterion = nn.CrossEntropyLoss() # Binary Cross Entropy loss

  L_hist = [] # loss history
  grad_hist = [] # gradient history
  progressbar = tqdm(total=N_TRAIN*EPOCH)
  data_processed = 0

  model.train() # train mode로 설정
  for epoch in range(EPOCH):

    rloss = 0 # running loss

    for x_batch, y_batch in train_DL:
      x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
      # print(x_batch.shape, y_batch.shape)

      # 1. inference
      y_hat = model(x_batch)
      # print(y_hat.shape)

      # 2. loss
      loss = criterion(y_hat, y_batch) # loss
      rloss += (loss.item() * x_batch.size(0))

      # 3. gradient
      optimizer.zero_grad() # optimizer
      loss.backward()
      # grad_hist.append(torch.sum(torch.abs(model.linear[0].weight.grad)).item())

      # 4. update weights
      optimizer.step()

      data_processed += x_batch.size(0)
      if data_processed > (N_TRAIN//10):
        progressbar.update(data_processed)
        progressbar.set_description(f'Epoch: {epoch+1}/{EPOCH}, Loss: {loss:.4f}')
        data_processed = 0

    rloss /= N_TRAIN
    L_hist.append(rloss)
    # print(f'Epoch: {epoch+1}, Loss: {rloss}')

    if data_processed > 0:
      progressbar.update(data_processed)
      progressbar.set_description(f'Epoch: {epoch+1}/{EPOCH}, Loss: {loss:.4f}')
      data_processed = 0

  L_hist_str = [f'{l:.5f}' for l in L_hist]
  print()
  print(L_hist_str)
  # print(grad_hist)
    
if TRAIN_MODEL:
  plt.figure(figsize=(6,4))
  plt.title(f'Loss ( {DS_NAME}, {MODEL_NAME}, EPOCH={EPOCH}, N_BATCH={N_BATCH}, LR={LR} )')
  # plt.subplot(1, 2, 1)
  plt.plot([i+1 for i in range(EPOCH)], L_hist, 'bs--', label='loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.grid()
  # plt.subplot(1, 2, 2)
  # # plt.plot([i+1 for i in range(EPOCH)], grad_hist, 'go--', label='gradient')
  # plt.xlabel('epoch')
  # plt.ylabel('gradient')
  # plt.legend()
  # plt.grid()
  
if TRAIN_MODEL and SAVE_MODEL:
  torch.save(model.state_dict(), MODEL_PATH + '/' + MODEL_NAME + '_' + DS_NAME + '.pt')
  print('Model saved.')


# Test
print_title('Test Section')

if LOAD_MODEL:
  model = globals()[MODEL_NAME](N_CH, N_WIDTH, N_HEIGHT, N_CLASSES).to(DEVICE)
  model.load_state_dict(torch.load(MODEL_PATH + '/' + MODEL_NAME + '_' + DS_NAME + '.pt'))
  print('Model loaded.')

  # 로드 확인
  print(model)
  # print(model.state_dict())
  
  
plt.figure(figsize=(15, 12))
subplot_index = 1
subplot_row = 6
subplot_col = 6
subplot_count = subplot_row * subplot_col

model.eval() # drop out, batch normalization 등 사용되었다면, eval() 모드와 train() 모드에서 결과 다르다.
with torch.no_grad(): # gradient 계산 중지
  correct = 0
  confusion = torch.zeros(N_CLASSES, N_CLASSES)
  for x_test, y_test in test_DL:
    x_test = x_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    y_hat = model(x_test)
    pred = y_hat.argmax(dim=1)
    # print(y_hat.shape)
    # print(pred)
    # print(y_test)
    # print(pred == y_test)
    # print(torch.sum(pred == y_test).item())
    # break

    correct += torch.sum(pred == y_test).item()
    confusion += torch.bincount(N_CLASSES * y_test + pred, minlength=N_CLASSES**2).cpu().reshape(N_CLASSES, N_CLASSES)

    # show incorrect pred
    incorrect_index = torch.nonzero(pred != y_test).reshape(-1)
    for i in incorrect_index:
      if subplot_count < subplot_index:
        break
      plt.subplot(subplot_row, subplot_col, subplot_index)
      if N_CH == 1:
        plt.imshow(x_test[i].cpu().squeeze(), cmap='gray')
      else:
        plt.imshow(x_test[i].cpu().squeeze().permute(1, 2, 0))
      pred_classname = train_DS.classes[pred[i].item()]
      y_test_classname = train_DS.classes[y_test[i].item()]
      plt.title(f'{pred_classname} ({y_test_classname})', color=('g' if pred[i] == y_test[i] else 'r'))
      plt.axis('off')
      subplot_index += 1

print(f'Test accuracy : {(correct/N_TEST)*100 :.2f}%, ({correct}/{N_TEST})')


# Show Confusion Matrix
plt.figure(figsize=(8, 8))
plt.title(f'Confusion Matrix ( {DS_NAME}, {MODEL_NAME}, EPOCH={EPOCH}, N_BATCH={N_BATCH}, LR={LR} )')
plt.imshow(confusion, cmap='Blues')
plt.xlabel(f'Predicted (accuracy : {(correct/N_TEST)*100 :.2f}%, ({correct}/{N_TEST}))')
plt.ylabel('True')
plt.xticks(range(N_CLASSES), train_DS.classes, rotation=45)
plt.yticks(range(N_CLASSES), train_DS.classes)
# plt.colorbar()
for i in range(N_CLASSES):
  for j in range(N_CLASSES):
    plt.text(j, i, int(confusion[i, j].item()), ha='center', va='center',
             color='w' if confusion[i, j].item() > confusion.max() / 2 else 'k')


plt.show()