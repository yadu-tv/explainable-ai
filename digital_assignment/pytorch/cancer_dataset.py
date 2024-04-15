import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import os
from dotenv import load_dotenv
from PIL import Image

class BreastCancerDataset(torch.utils.data.Dataset):
    def __init__(self,paths): #Defines the path
        self.paths=paths
    
    def __len__(self): #Gets number of images
        return len(self.paths)
    
    def __getitem__(self,index): #Gets each image from paths
        sample=self.paths[index]
        
        load_dotenv('.env')
        x_env_path=os.getenv("Images")
        y_env_path=os.getenv("Mask")
        
        x_path,y_path=sample[0],sample[1]
        
        #Read the image
        x_image,y_image=Image.open(x_env_path+x_path),Image.open(y_env_path+y_path)
        
        #Preprocessing -> Convert to Tensor
        process_function_x=T.Compose([
            T.Resize((512,512)),T.ToTensor(),T.Normalize(
                mean=(0.5,0.5,0.05),std=(0.5,0.5,0.5)
            )
        ])
        
        process_function_y=T.Compose([
            T.Resize((512,512)),T.ToTensor()
        ])
        
        x_tensor=process_function_x(x_image)
        y_tensor=process_function_y(y_image)
        
        return x_tensor, y_tensor