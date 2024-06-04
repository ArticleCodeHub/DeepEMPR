import os
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.utils.data as data
from torchvision.transforms import InterpolationMode 
from torchvision.models import AlexNet_Weights,VGG16_Weights,ResNet50_Weights
import torchvision
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
from PIL import Image
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sn
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.utils import save_image
from sklearn.metrics import f1_score
import numpy.matlib
from sklearn.preprocessing import label_binarize

def HDMR0(I):
    [n1,n2, n3] = I.shape
    f0 = np.sum(I) /(n1*n2*n3)
    return f0

def HDMR1(I):
    [n1, n2, n3] = I.shape
    f0 = HDMR0(I)
    
    f1 = np.sum(I,axis = 1)/(n2*n3)
    f2 = (np.sum(I,axis = 0)/(n1*n3)).transpose(1,0)
    f3 = np.sum(I)/(n1*n2)
    
    f1 = np.matlib.repmat(f1, 1, n2) - f0
    f2 = np.matlib.repmat(f2, n1, 1) - f0
    f3 = np.matlib.repmat(f3, n1, n2) - f0
    return f1, f2, f3

def HDMR2(I):
    [n1, n2, n3] = I.shape
    f0 = HDMR0(I)
    [f1, f2, f3] = HDMR1(I)
    
    f12 = np.sum(I,axis=2) / n3
    f13 = np.sum(I,axis=1) / n2
    f23 = (np.sum(I,axis=0) / n1).transpose(1,0)
    
    f12 = np.matlib.repmat(f12, 1, 1) - (f1 + f2 + f0)
    f13 = np.matlib.repmat(f13, 1, n2) - (f1 + f3 + f0)
    f23 = np.matlib.repmat(f23, n1, 1) - (f2 + f3 + f0)
    
    return f12, f13, f23

def approx(I):
    [n1, n2, n3] = I.shape
    f0 = HDMR0(I)
    [f1, f2, f3] = HDMR1(I)
    [f12, f13, f23] = HDMR2(I)
    
    Recdreccons = f0 + f1 + f2 + f3 + f12 + f13 + f23
    Recdiki = f12 + f13 + f23
    Recdbir = f1 + f2 + f3
    
    return Recdreccons + 2*(Recdbir + Recdiki)

def EMPR0(s1, s2, s3, I):
    [n1,n2,n3] = I.shape
    f0 = 0
    tmp_arr = np.outer(s1,s2)
    
    f0 = sum(sum((np.ones((n1,n2))*s3[0]*tmp_arr)*I[:,:,0]))
    f0 = f0 + sum(sum((np.ones((n1,n2))*s3[1]*tmp_arr)*I[:,:,1]))
    f0 = f0 + sum(sum((np.ones((n1,n2))*s3[2]*tmp_arr)*I[:,:,2]))
    f0 = f0/(n1*n2*n3)
    
    return f0

def EMPR1(f0, s1, s2, s3, I):  
    [n1,n2,n3] = I.shape
    
    f1 = sum(((s3[0]*np.ones((n1,n2)))*s2*I[:,:,0]).transpose())
    f1 = f1 + sum(((s3[1]*np.ones((n1,n2)))*s2*I[:,:,1]).transpose())
    f1 = f1 + sum(((s3[2]*np.ones((n1,n2)))*s2*I[:,:,2]).transpose())
    f1 = f1 /(n2*n3) - f0*s1
    
    f2 = sum(((np.ones((n2,n1))*s1).transpose()*s3[0]*I[:,:,0]))
    f2 = f2 + sum(((np.ones((n2,n1))*s1).transpose()*s3[1]*I[:,:,1]))
    f2 = f2 + sum(((np.ones((n2,n1))*s1).transpose()*s3[2]*I[:,:,2]))
    f2 = f2 /(n1*n3) - f0*s2
    
 
    temp = np.stack([np.outer(s1,s2), np.outer(s1,s2), np.outer(s1,s2)]).transpose(1,2,0)
    f3 = sum(sum(temp*I));
    f3 = f3/(n1*n2) -f0*s3
       
    return f1, f2, f3

def EMPR2(f0, f1, f2, f3, s1, s2, s3, I):
    [n1,n2,n3] = I.shape
    
    
    f12 = (I[:,:,0]*s3[0] + I[:,:,1]*s3[1] + I[:,:,2]*s3[2])/n3;
    f12 = f12 - f0 * np.outer(s1,s2) - np.outer(f1,s2) - np.outer(s1,f2)
    
    tmp = np.stack([np.ones((n1,n2))*s2, np.ones((n1,n2))*s2, np.ones((n1,n2))*s2]).transpose(1,2,0)
    f13 = sum((tmp*I).transpose(1,0,2))/n2
    f13 = f13 - f0 * np.outer(s1,s3) - np.outer(f1,s3) - np.outer(s1,f3)
    
    tmp = np.stack([np.ones((n2,n1))*s1, np.ones((n2,n1))*s1, np.ones((n2,n1))*s1]).transpose(2,1,0)
    f23 = sum((tmp*I))/n1
    f23 = f23 - f0 * np.outer(s2,s3) - np.outer(f2,s3) - np.outer(s2,f3)
        
    return f12, f13, f23

def SecondaryApproximation(s1, s2, s3, I):
    [n1,n2,n3] = I.shape
    
    f0 = EMPR0(s1, s2, s3, I)
    [f1, f2, f3] = EMPR1(f0, s1, s2, s3, I)
    [f12, f13, f23] = EMPR2(f0, f1, f2, f3, s1, s2, s3, I)
    
    
    f0_term = np.stack([f0*np.outer(s1,s2)*s3[0], f0*np.outer(s1,s2)*s3[1], f0*np.outer(s1,s2)*s3[2]]).transpose(1,2,0)
    f1_term = np.stack([np.outer(f1,s2)*s3[0], np.outer(f1,s2)*s3[1], np.outer(f1,s2)*s3[2]]).transpose(1,2,0)
    f2_term = np.stack([np.outer(s1,f2)*s3[0], np.outer(s1,f2)*s3[1], np.outer(s1,f2)*s3[2]]).transpose(1,2,0)
    f3_term = np.stack([np.outer(s1,s2)*f3[0], np.outer(s1,s2)*f3[1], np.outer(s1,s2)*f3[2]]).transpose(1,2,0)
    f12_term = np.stack([f12*s3[0], f12*s3[1], f12*s3[2]]).transpose(1,2,0)
    f23_term = (np.outer(f23,s1.reshape(1,1,n1)).reshape(n2,n3,n1)).transpose(2,0,1)
    f13_term = (np.outer(f13,s2.reshape(1,1,n2)).reshape(n1,n3,n2)).transpose(0,2,1)
    
    
    newII = I - (f0_term + f1_term + f2_term + f3_term + f12_term + f23_term + f13_term)
    newI = newII + f3_term + f23_term + f13_term   
    
    return newI

def supports(I):
    [n1,n2,n3] = I.shape
    
    s1 = (sum(sum(np.double(I.transpose(1,2,0)))))/(n2*n3)
    s1 = s1 / np.sqrt(sum((1/n1)*(s1**2)))
    
    s2 = (sum(sum(np.double(I.transpose(0,2,1)))))/(n1*n3)
    s2 = s2 / np.sqrt(sum((1/n2)*(s2**2)))
    
    s3 = (sum(sum(np.double(I))))/(n1*n2)
    s3 = s3 / np.sqrt(sum((1/n3)*(s3**2)))

    return s1, s2, s3

def init_weights(m):
     if type(m) == nn.Conv2d or type(m) == nn.Linear:
         nn.init.xavier_uniform_(m.weight)
         if m.bias is not None:
             m.bias.data.fill_(0.01)

def read_four_channel_image(path):
    with open(path, 'rb') as f:
        st = time.time()
        img = Image.open(f)
        transform = transforms.ToTensor()
        tensor = transform(img)
        ndarray = tensor.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
        ndarray[:,:,[0,2]] = ndarray[:,:,[2,0]]# rgb<->bgr

    
        [s1, s2, s3] = supports(np.double(ndarray))
        IG_tmp = SecondaryApproximation(s1, s2, s3, np.double(ndarray))
        IG_tmp = 255*(IG_tmp - np.min(IG_tmp)) /(np.max(IG_tmp) - np.min(IG_tmp))
        IG = (0.2989 * IG_tmp[:,:,0] + 0.5870 * IG_tmp[:,:,1] + 0.1140 * IG_tmp[:,:,2]).reshape(ndarray.shape[0],ndarray.shape[1],1)
        IG_eq = approx(IG)
        
        Inew=np.zeros((ndarray.shape[0],ndarray.shape[1], 4), dtype=np.uint8)
        ndarray[:,:,[2,0]] = ndarray[:,:,[0,2]]
        Inew[:,:,0:3] = ndarray
        Inew[:,:,3] = np.clip(IG_eq, 0, 255).astype(np.uint8) #IG_eq #
        
        tensor_image = torch.from_numpy(Inew).transpose(0,2) #reshape(Inew.shape[2],Inew.shape[0],Inew.shape[1])
        tensor_image = tensor_image.transpose(1,2)
        tensor_image = tensor_image.float() / 255.0
        to_pil = transforms.ToPILImage()
        pil_img = to_pil(tensor_image)
        end = time.time()
        return pil_img

def mean_std(loader):
  mean_all = 0
  std_all = 0
  count = 0
  for images, labels in loader:
      # shape of images = [b,c,w,h]
      mean, std = images.mean([0,2,3]), images.std([0,2,3])
      aa = images
      mean_all = mean_all + mean
      std_all = std_all + std
      count = count + 1
  return aa,mean_all/count, std_all/count

#we define hyperparameters
NUM_EPOCHS = 30
BATCH_SIZE = 32
LRATE = 0.001
MOMENTUM = 0.9
DECAY = 0.0005


TRAIN_DATA_PATH = 'C:\\Users\\lenovo\\Desktop\\Yeni klasör\\Makale Veri Araştırma\\dataset\\LeafSeverity\\train'
VAL_DATA_PATH = 'C:\\Users\\lenovo\\Desktop\\Yeni klasör\\Makale Veri Araştırma\\dataset\\LeafSeverity\\val'
TEST_DATA_PATH = 'C:\\Users\\lenovo\\Desktop\\Yeni klasör\\Makale Veri Araştırma\\dataset\\LeafSeverity\\test'   

transform_temp = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ]) 

train_data_meanstd = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform_temp, loader = read_four_channel_image)
train_data_loader_meanstd = data.DataLoader(train_data_meanstd)
aa, mean, std = mean_std(train_data_loader_meanstd)

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomApply([transforms.RandomRotation(10)], 0.25),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
    transforms.ToTensor(),
    #transforms.Grayscale(num_output_channels=1),
    #transforms.Normalize([0.5, ], [0.5, ]),
    #transforms.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5]),
    transforms.Normalize(list(mean.numpy()), list(std.numpy())),
    ]) 

transform_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Grayscale(num_output_channels=1),
    #transforms.Normalize([0.5, ], [0.5, ]),
    #transforms.Normalize([0.485, 0.456, 0.406, 0.5], [0.229, 0.224, 0.225, 0.5]),
    transforms.Normalize(list(mean.numpy()), list(std.numpy())),
    ]) 

#Train, Test and Validation set
train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=transform_train, loader = read_four_channel_image)
train_data_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

val_data = torchvision.datasets.ImageFolder(root=VAL_DATA_PATH, transform=transform_val_test, loader = read_four_channel_image)
val_data_loader = data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=transform_val_test, loader = read_four_channel_image)
test_data_loader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

dataloaders = {'train': train_data_loader, 'val': val_data_loader, 'test': test_data_loader}

#MODELS
#################################### ALEXNET ################################################

# #model = models.alexnet(weights = None)
# model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1) #pretrained AlexNet
# input_features = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(in_features=input_features, out_features=5)
# #model.apply(init_weights) # burada xavier ağırlıklarıyla ilklendiriyoruz.

# ## Multichannel(4-channel) görüntüler için ilk katman tasarımı
# w = model.features[0].weight.clone()
# model.features[0] = nn.Conv2d(4, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
# with torch.no_grad():
#     model.features[0].weight.data[:,0:3,:,:] = w.data
#     model.features[0].weight.data[:,3,:,:] = (w.data[:,0,:,:] + w.data[:,1,:,:] + w.data[:,2,:,:])/3.0  
#     #model.features[0].weight.data[:,3,:,:] = 0.2989*w.data[:,0,:,:] + 0.5870*w.data[:,1,:,:] + 0.1140*w.data[:,2,:,:] 

#################################### VGG16 ###############################################   


# model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1) #pretrained VGG16
# input_features = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(in_features=input_features, out_features=5)

# ## Multichannel(4-channel) görüntüler için ilk katman tasarımı
# w = model.features[0].weight.clone()
# model.features[0] = nn.Conv2d(4, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
# with torch.no_grad():
#     model.features[0].weight.data[:,0:3,:,:] = w.data
#     model.features[0].weight.data[:,3,:,:] = (w.data[:,0,:,:] + w.data[:,1,:,:] + w.data[:,2,:,:])/3.0
#     #model.features[0].weight.data[:,3,:,:] = 0.2989*w.data[:,0,:,:] + 0.5870*w.data[:,1,:,:] + 0.1140*w.data[:,2,:,:] 

###################################### RESNET-50 #############################################


model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1) #pretrained ResNet50
input_features = model.fc.in_features
model.fc = nn.Linear(in_features=input_features, out_features=5)

## Multichannel(4-channel) görüntüler için ilk katman tasarımı
w = model.conv1.weight.clone()
model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
with torch.no_grad():
    model.conv1.weight.data[:,0:3,:,:] = w.data
    model.conv1.weight.data[:,3,:,:] = (w.data[:,0,:,:] + w.data[:,1,:,:] + w.data[:,2,:,:])/3.0
    #model.features[0].weight.data[:,3,:,:] = 0.2989*w.data[:,0,:,:] + 0.5870*w.data[:,1,:,:] + 0.1140*w.data[:,2,:,:] 

###################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LRATE, momentum=MOMENTUM, weight_decay=DECAY)

# Define a learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

best_acc = 0.0
best_model_acc = copy.deepcopy(model.state_dict())

best_loss = 100.0
best_model_loss = copy.deepcopy(model.state_dict())

model_array = []
for epoch in range(NUM_EPOCHS):
    
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train() #ağırlık güncellemesi var 
        else:
            model.eval() #ağırlık güncellemsi yok
        
        running_loss = 0.0
        correct_predictions = 0
        
        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
            
            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels.data)
        
        scheduler.step()
        
        epoch_loss = running_loss / len(dataloaders[phase].dataset)
        epoch_acc = correct_predictions.double() / len(dataloaders[phase].dataset)
        
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_acc = copy.deepcopy(model.state_dict())
        
        if phase == 'val' and epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_loss = copy.deepcopy(model.state_dict())
        
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    aModel = copy.deepcopy(model.state_dict())
    model_array.append(aModel)
    print("-----------------------------------------------------------------------")

###########################################################################################
# Evaluating the model on testing set for all epoch 
for i in range(len(model_array)):#
    model.load_state_dict(model_array[i])
    model.eval()
    
    pred_all =  []
    labels_all = []
    outputs_all = []
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        preds = preds.data.cpu().numpy()
        pred_all.extend(preds)
        outputs_all.extend(torch.nn.functional.softmax(outputs, dim=1))
        
        labels = labels.data.cpu().numpy()
        labels_all.extend(labels)
        
    #creating confusion matrix
    conf = confusion_matrix(labels_all, pred_all)
    print("Accuracy: ",(conf[0][0]+conf[1][1]+conf[2][2]+conf[3][3]+conf[4][4])/253.0)
    confusion_matrix_plot = pd.DataFrame(conf, range(5), range(5))
    
    # confusion_matrix_plot.columns = ['brown', 'cercospora', 'healthy','miner', 'rust' ]
    # confusion_matrix_plot.index = ['brown', 'cercospora', 'healthy','miner', 'rust' ]
    confusion_matrix_plot.columns = ['healthy', 'high', 'low','vhigh', 'vlow' ]
    confusion_matrix_plot.index = ['healthy', 'high', 'low','vhigh', 'vlow' ]
    
    sn.heatmap(confusion_matrix_plot, annot=True, annot_kws={"size": 7})
    print(classification_report(labels_all, pred_all, target_names=['healthy', 'high', 'low','vhigh', 'vlow' ]))
############################################################################################
###########################################################################################
# Evaluating the model for the epoch that has the best model accuracy or best model loss.

#model.load_state_dict(best_model_loss)
model.load_state_dict(best_model_acc)
model.eval()
    
pred_all =  []
labels_all = []
outputs_all = []
for inputs, labels in dataloaders['test']:
    inputs = inputs.to(device)
    labels = labels.to(device)
        
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
        
    preds = preds.data.cpu().numpy()
    pred_all.extend(preds)
    outputs_all.extend(torch.nn.functional.softmax(outputs, dim=1))
        
    labels = labels.data.cpu().numpy()
    labels_all.extend(labels)
        
#creating confusion matrix
conf = confusion_matrix(labels_all, pred_all)
print("Accuracy: ",(conf[0][0]+conf[1][1]+conf[2][2]+conf[3][3]+conf[4][4])/253.0)
confusion_matrix_plot = pd.DataFrame(conf, range(5), range(5))
    
# confusion_matrix_plot.columns = ['brown', 'cercospora', 'healthy','miner', 'rust' ]
# confusion_matrix_plot.index = ['brown', 'cercospora', 'healthy','miner', 'rust' ]
confusion_matrix_plot.columns = ['healthy', 'high', 'low','vhigh', 'vlow' ]
confusion_matrix_plot.index = ['healthy', 'high', 'low','vhigh', 'vlow' ]
  
sn.heatmap(confusion_matrix_plot, annot=True, annot_kws={"size": 7})
print(classification_report(labels_all, pred_all, target_names=['healthy', 'high', 'low','vhigh', 'vlow' ]))
###########################################################################################

############################# ROC-AUC for best #################################
num_classes = 5
name_classes = ['healthy', 'high', 'low','vhigh', 'vlow' ]
# Binarize the true labels
y_true_bin = label_binarize(labels_all, classes=np.arange(num_classes))
y_pred_prob = np.array([outputs_all[i].detach().numpy() for i in range(len(outputs_all)) ])

# Compute ROC curve and ROC-AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):  # Assuming 3 classes
     fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
     roc_auc[i] = auc(fpr[i], tpr[i])
        
# Plot the ROC-AUC curves for each class
plt.figure(figsize=(8, 6))
colors = ['b', 'g', 'r','m','c']  # You can choose different colors for each class
for i in range(5):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label=f'Class {name_classes[i]} (ROC-AUC = {roc_auc[i]:.4f})')
    
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Plot the diagonal line for reference
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend(loc="lower right")
plt.show()
