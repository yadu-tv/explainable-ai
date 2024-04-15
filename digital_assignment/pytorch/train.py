import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import cuda, cpu
import gc
from sklearn.model_selection import train_test_split
from cancer_dataset import BreastCancerDataset
from dotenv import load_dotenv
from Model.model import CombinedModel
import os

def train_step():
    epoch_loss=0
    
    for step,(x_sample,y_sample) in enumerate(train_loader):
        x_sample=x_sample.to(device=device)
        y_sample=y_sample.to(device=device)
        
        #Predictions - forward propogation
        predictions=model(x_sample)
        
        #Backpropogation
        model.zero_grad()
        
        loss_value=loss(predictions,y_sample)
        
        loss_value.backward()
        model_optimizer.step()
        
        #Add losses
        epoch_loss+=loss_value.item()
        
        #Preserve memory
        del x_sample
        del y_sample
        del predictions
        
        cuda.empty_cache()
    
    return epoch_loss/train_steps

def test_step():
    epoch_loss=0
    
    for step,(x_sample,y_sample) in enumerate(test_loader):
        x_sample=x_sample.to(device=device)
        y_sample=y_sample.to(device=device)
        
        #Predictions - forward propogation
        predictions=model(x_sample)
        
        loss_value=loss(predictions,y_sample)
        
        #Add losses
        epoch_loss+=loss_value.item()
        
        #Preserve memory
        del x_sample
        del y_sample
        del predictions
        
        cuda.empty_cache()
    
    return epoch_loss/test_steps

def training_loop():
    for epoch in range(epochs):
        model.train(True) #move model to training mode
        
        train_loss=train_step()
        model.eval()
        
        with torch.no_grad():
            test_loss=test_step()
            
            print("Epoch: ", epoch+1)
            print("Train loss: ",train_loss)
            print("Test loss: ",test_loss)
            
            #checkpoints
            if((epoch+1) % 10 == 0):
                torch.save(model.state_dict(),
                           'weights/model{epoch}.pth'.format(epoch=epoch+1))

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system') #Ensures that thereis no file limit while processing dataset
    
    load_dotenv('.env') #Path to env file
    image_env_paths=os.getenv("Images")
    mask_env_paths=os.getenv("Mask")
    
    #Get data, batch data, run models
    
    #Get Data - Dataset
    #Batch Data - Dataloader
    #Run Data - Models
    
    params = {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 0
    }
    
    #Store paths in a list
    image_paths=os.getenv("Images")
    mask_paths=os.getenv("Mask")
    
    #Get the tif files
    image_paths=os.listdir(image_paths)
    mask_paths=os.listdir(mask_paths)
    
    #Filter out fragmented files
    image_paths_new=list()
    mask_paths_new=list()
    
    for i in image_paths:
        if i[1] not in ['_'] and '.xml' not in i:
            image_paths_new.append(i)
    for j in mask_paths:
        if j[1] not in ['_'] and '.xml' not in j:
            mask_paths_new.append(j)
            
    image_paths_new=sorted(image_paths_new)
    mask_paths_new=sorted(mask_paths_new)
            
    #Map the paths as X and Y
    paths_dataset=list(
        zip(image_paths_new,mask_paths_new)
    )
    
    #Train-Test split
    train,test=train_test_split(paths_dataset, train_size=0.85)
    
    #Call the dataset
    train_set=BreastCancerDataset(train)
    test_set=BreastCancerDataset(test)
    
    #Create a Dataloader
    train_loader=DataLoader(dataset=train_set,**params)
    test_loader=DataLoader(dataset=test_set,**params)
    
    #Device
    device=torch.device("cuda")
    
    #Hyperparameters
    lr=0.001
    epochs=100
    
    model=CombinedModel().to(device=device)
    model_optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.999),weight_decay=0.001)
    
    #Loss function
    loss=nn.BCEWithLogitsLoss() #more stable than BCE loss
    
    train_steps=(len(train_set)+params['batch_size'])//params['batch_size']
    test_steps=(len(test_set)+params['batch_size'])//params['batch_size']
    
    training_loop()