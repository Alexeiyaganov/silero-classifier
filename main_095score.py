import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path


# Read metadata file
metadata_file = "train.csv"
df = pd.read_csv(metadata_file)


# Take relevant columns
df = df[['wav_path', 'target']]
import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio

from torch.utils.data import DataLoader, Dataset, random_split

import pandas as pd
import os

import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init

torch.set_num_threads(4)

class AudioUtil():
  # Загружаем аудиофайл и возвращаем тензор
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
    # Nothing to do
            return aud

        num_channels = sig.shape[0]
    # Resample first channel
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        if (num_channels > 1):
    # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))

  # ----------------------------
  # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
  # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
          # Truncate the signal to the given length
          sig = sig[:,:max_len]

        elif (sig_len < max_len):
          # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)
    
    
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)


  # ----------------------------
  # Generate a Spectrogram
  # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

  # ----------------------------
  # Augment the Spectrogram by masking out some sections of it in both the frequency
  # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
  # overfitting and to help the model generalise better. The masked sections are
  # replaced with the mean value.
  # ----------------------------
@staticmethod
def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
    def __init__(self, df, data_path):
    
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # Длина датасета
    def __len__(self):
        return len(self.df)    
    
  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
    def __getitem__(self, idx):
        # путь до файла
        audio_file = self.data_path + self.df.loc[idx, 'wav_path']
        # Берем класс
        class_id = self.df.loc[idx, 'target']

        aud = AudioUtil.open(audio_file)
        reaud = AudioUtil.resample(aud, self.sr)

        dur_aud = AudioUtil.pad_trunc(reaud, self.duration)
        
        #add noises
        shi_aud=AudioUtil.time_shift(dur_aud,self.shift_pct)  #нужно установить размер шумов
        
        sgram = AudioUtil.spectro_gram(shi_aud, n_mels=64, n_fft=1024, hop_len=None)

        return sgram, class_id, self.df.loc[idx, 'wav_path']


# make your path
data_path = "train/"
myds = SoundDS(df, data_path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

# ----------------------------

class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]
        # Second Convolution Block
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]
        # Third Convolution Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]
        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=3)
        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
#         print(x.shape)
        x = self.conv(x)
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        # Linear layer
        x = self.lin(x)
        # Final output
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# ----------------------------
# Training
# ----------------------------
def training(model, train_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            #if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        print('Finished Training')
  
num_epochs=1   # Because of long time of training.
training(myModel, train_dl, num_epochs)
torch.save(myModel, 'model_after.pth')


from sklearn.metrics import confusion_matrix
# ----------------------------
# Inference
# ----------------------------
def inference (model, val_dl):
    correct_prediction = 0
    total_prediction = 0

  # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
          # Get the input features and target labels, and put them on the GPU
          inputs, labels = data[0].to(device), data[1].to(device)

          # Normalize the inputs
          inputs_m, inputs_s = inputs.mean(), inputs.std()
          inputs = (inputs - inputs_m) / inputs_s

          # Get predictions
          outputs = model(inputs)
          print(outputs)

          # Get the predicted class with the highest score
          _, prediction = torch.max(outputs,1)
            
        
            
          # Count of predictions that matched the target label
          correct_prediction += (prediction == labels).sum().item()
          total_prediction += prediction.shape[0]
          
        
        
    
    acc = correct_prediction/total_prediction
    res=confusion_matrix(labels, prediction)
    print(res)
    
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')
# val_dlned model with the validation set
inference(myModel, val_dl)



metadata_file_val = "./sample_submission.csv"
df_val = pd.read_csv(metadata_file_val)


# Take relevant columns
df_val = df_val[['wav_path', 'target']]

data_path_test = "val/"
myds_test = SoundDS(df1, data_path_test)


# Create training and validation data loaders

test_dl = torch.utils.data.DataLoader(myds_test, batch_size=16, shuffle=False)

from sklearn.metrics import confusion_matrix
import numpy
# ----------------------------
# Inference
# ----------------------------
def inference_test (model, val_dl):
    data_empty = {'wav_path': [], 'target': []}
    df_res = pd.DataFrame(data_empty)

  # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
          # Get the input features and target labels, and put them on the GP2
          inputs, labels = data[0].to(device), data[1].to(device)
          # print(data[2])
          # print(data[1].to(device))

          # Normalize the inputs
          inputs_m, inputs_s = inputs.mean(), inputs.std()
          inputs = (inputs - inputs_m) / inputs_s

          # Get predictions
          outputs = model(inputs)
#         #  print(outputs)

          # Get the predicted class with the highest score
          _, prediction = torch.max(outputs,1)
          prediction=prediction.numpy()
          for i in range(len(data[2])):
          	row=[data[2][i],int(prediction[i])]
		df_res.loc[len(df_res)] = row
          

    df_res.to_csv("./submission.csv",encoding='utf-8', index=False)

myModel = torch.load("./model_after.pth")
inference_test(myModel, test_dl)


