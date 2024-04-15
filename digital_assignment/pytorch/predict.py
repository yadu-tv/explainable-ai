import torch
import numpy as np
import matplotlib.pyplot as plt
from Model.model import CombinedModel
from dotenv import load_dotenv
import os
import random
import torchvision.transforms as T
from PIL import Image
import cv2
import torch.nn.functional as f

if __name__ == '__main__':
    weights=torch.load("weights/model100.pth")
    
    load_dotenv('.env')
    
    model=CombinedModel()
    model.eval()
    
    #Load model with trained weights
    model.load_state_dict(weights)
    
    path=os.getenv("Images")
    xpaths=sorted(os.listdir(path))
    
    #Remove fragments
    xps=list()
    for i in xpaths:
        if i[1] not in ['_'] and '.xml' not in i:
            xps.append(i)
            
    #select any random image
    random_image_path=random.choice(xps)
    
    image=Image.open(path+random_image_path)
    tens_transform=T.Compose([
        T.Resize((512,512)), T.ToTensor(), T.Normalize(
                mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)
        )
    ])
    
    image_tensor=tens_transform(image)
    image_tensor=image_tensor.view(1,3,512,512)
    
    #Get outputs
    predictions=model(image_tensor)
    predictions_probabs=f.sigmoid(predictions)
    
    #Map
    mask_pred=torch.where(predictions_probabs>0.5,1.0,0.0)
    
    #detach mask from gpu and convert to numpy
    mask_np=mask_pred.detach().numpy()
    mask_np=mask_np.astype(np.uint8)
    
    mask_final=mask_np.transpose(0,2,3,1)
    
    #save the prediction
    plt.imshow(mask_final[0][:][:][0])