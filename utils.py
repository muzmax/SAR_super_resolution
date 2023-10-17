import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from scipy import special
from scipy import signal
import numpy as np
import cv2
import pathlib
import os
import glob

from models.SwinTransformer import SwinIR
from models.SRCNN import SRCNN
from models.PixelShuffle import PixelShuffle

def get_model(cfg,model_name,device='cuda'):
    model_name = model_name.lower()
    if model_name == "swintransformer":
        model = SwinIR(
            upscale=cfg["DATASET"]["PREPROCESSING"]["DOWNSCALE_FACTOR"],
            in_chans=cfg["MODEL"]["SWINTRANSFORMER"]["IN_CHANNELS"],
            img_size=cfg["MODEL"]["SWINTRANSFORMER"]["IMG_SIZE"],
            window_size=cfg["MODEL"]["SWINTRANSFORMER"]["WINDOW_SIZE"],
            img_range=cfg["MODEL"]["SWINTRANSFORMER"]["IMG_RANGE"],
            depths=cfg["MODEL"]["SWINTRANSFORMER"]["DEPTHS"],
            embed_dim=cfg["MODEL"]["SWINTRANSFORMER"]["EMBED_DIM"],
            num_heads=cfg["MODEL"]["SWINTRANSFORMER"]["NUM_HEADS"],
            mlp_ratio=cfg["MODEL"]["SWINTRANSFORMER"]["MLP_RATIO"],
            upsampler=cfg["MODEL"]["SWINTRANSFORMER"]["UPSAMPLER"],
            resi_connection=cfg["MODEL"]["SWINTRANSFORMER"]["RESI_CONNECTION"],)
        model = model.to(device)
        model = model.eval()
        return model

    elif model_name == "srcnn":
        model = SRCNN(cfg)
        model = model.to(device)
        model = model.eval()
        return model
    elif model_name == "pixelshuffle":
        model = PixelShuffle(cfg)
        model = model.to(device)
        model = model.eval()
        return model
    elif model_name == "bicubic":
        model = torch.nn.Upsample(scale_factor=cfg["DATASET"]["PREPROCESSING"]["DOWNSCALE_FACTOR"], mode="bicubic")
        model.to(device)
        return model
    else:
        return None
    

def load_network(model, load_path, pretrained=False, strict=True, param_key="params"):
    """Function to load pretrained model or checkpoint
    Args:
        load_path (string): the path of the checkpoint to load
        model (torch.nn.module): the network
        strict (bool, optional): If the model is strictly the same as the one we load. Defaults to True.
        param_key (str, optional): the key inside state dict. Defaults to 'params'.
    """
    if strict:
        state_dict = torch.load(load_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        model.load_state_dict(state_dict, strict=strict)
        del state_dict
    else:
        state_dict_old = torch.load(load_path)
        if param_key in state_dict_old.keys():
            state_dict_old = state_dict_old[param_key]
        # Compute weights mean of the first conv layer to go from 3 channel to 1 channel
        if pretrained and "conv_first.weight" in state_dict_old.keys():
            state_dict_old["conv_first.weight"] = torch.mean(
                state_dict_old["conv_first.weight"], 1, True
            )
        # Init New dict
        state_dict = model.state_dict()
        # Some weights cannot be processed because they depend on the input channel value
        for key, value in state_dict_old.items():
            if state_dict[key].shape == value.shape:
                state_dict.update({key: value})
        model.load_state_dict(state_dict, strict=strict)
        del state_dict_old, state_dict

class ToTensor():
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, im):  
        assert isinstance(im,np.ndarray)      
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im = im.transpose((2, 0, 1))
        return torch.from_numpy(im)

class denormalizeSAR():
    def __init__(self,thresh,mean,std) -> None:
        self.thresh = thresh
        self.mean = mean
        self.std = std
    def __call__(self, im:np.array):
        im = np.power(10,im*self.std+self.mean)-self.thresh
        return im

class normalizeSAR():
    def __init__(self,thresh,mean,std) -> None:
        self.thresh = thresh
        self.mean = mean
        self.std = std
    def __call__(self, im:torch.tensor):
        im = (torch.log10(torch.abs(im)+self.thresh)-self.mean)/self.std
        return im

class SARdataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.files_names = glob.glob(os.path.join(root,'*.npy'))
        self.transform = transform
        
    def __getitem__(self, idx):
        im = np.load(self.files_names[idx])
        assert len(im.shape) == 2
        im = im[:,:,np.newaxis]
        name = os.path.basename(self.files_names[idx])
        name = os.path.splitext(name)[0]
        return self.transform(im).float(), name

    def __len__(self):
        return len(self.files_names)

def load_test(cfg,im_path,norm_param):
    # Init test dataset
    transform = transforms.Compose([ToTensor(),normalizeSAR(norm_param[0],norm_param[1],norm_param[2])])

    test_data = SARdataset(im_path, transform)
    # Build Loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=cfg["INFERENCE"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"])
    return test_loader

def inference(im,model,device,cfg):
    upscale = cfg["DATASET"]["PREPROCESSING"]["DOWNSCALE_FACTOR"]
    # im = im[:,:,:512,:512] # a enlever après
    im = im.to(device)
    patch_size = cfg["DATASET"]["IMAGE_SIZE"] 
    stride = cfg["INFERENCE"]["STRIDE"] 
    (un,c,h,w) = im.shape
    result = torch.zeros(c,h*upscale,w*upscale,device=device)
    count = torch.zeros(h*upscale,w*upscale,device='cpu')
    if h == patch_size:
        x_range = list(np.array([0]))
    else:
        x_range = list(range(0,h-patch_size,stride))
        if (x_range[-1]+patch_size)<h : x_range.extend(range(h-patch_size,h-patch_size+1))
    if w == patch_size:
        y_range = list(np.array([0]))
    else:
        y_range = list(range(0,w-patch_size,stride))
        if (y_range[-1]+patch_size)<w : y_range.extend(range(w-patch_size,w-patch_size+1))
    for x in x_range:
        for y in y_range:
            out = model(im[:,:,x:x+patch_size,y:y+patch_size])
            result[:,x*upscale:(x+patch_size)*upscale,y*upscale:(y+patch_size)*upscale] += torch.squeeze(out)
            count[x*upscale:(x+patch_size)*upscale,y*upscale:(y+patch_size)*upscale] += torch.ones(patch_size*upscale,patch_size*upscale,device='cpu')
    result = torch.div(result.cpu(),count)
    return result

def normalize01(im,val=None):
    if val == None:
        m = np.amin(im)
        M = np.amax(im)
    else:
        m = val[0]
        M = val[1]
    im_norm = (im-m)/(M-m)
    return im_norm

# Apply a treshold, a defined treshold or mean+3*var
def tresh_im(img,treshold=None,k=3):
    imabs = np.abs(img)
    if treshold == None:
        mean = np.mean(imabs)
        std = np.std(imabs)
        treshold = mean+k*std
        imabs = np.clip(imabs,None,treshold)
        imabs = normalize01(imabs)
    else:
        imabs = np.clip(imabs,None,treshold)
        imabs = normalize01(imabs)
    return imabs

# Save an image
def save_im(im,fold,is_sar=True,tresh=None):
    im = np.abs(im)
    shape_im = im.shape
    assert len(shape_im) == 2 or (len(shape_im) == 3 and shape_im[2])
    if is_sar:
        im = tresh_im(im,treshold=tresh)*255
    else :
        im = normalize01(im)*255
    cv2.imwrite(fold, im)

def get_path_param(SR_type,cfg):
    if SR_type[0] == 'n':
        im_path = cfg["INFERENCE"]["PATH_TO_SLC"]
        thresh_lr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_LR"]["THRESHOLD"]
        mean_lr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_LR"]["MEAN"]
        std_lr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_LR"]["STD"]
    elif SR_type[0] == 'd':
        im_path = cfg["INFERENCE"]["PATH_TO_DENOISED"]
        thresh_lr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_LR_DN"]["THRESHOLD"]
        mean_lr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_LR_DN"]["MEAN"]
        std_lr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_LR_DN"]["STD"]
    if SR_type[-1] == 'n':
        thresh_hr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_HR"]["THRESHOLD"]
        mean_hr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_HR"]["MEAN"]
        std_hr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_HR"]["STD"]
    if SR_type[-1] == 'd':
        thresh_hr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_HR_DN"]["THRESHOLD"]
        mean_hr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_HR_DN"]["MEAN"]
        std_hr = cfg["DATASET"]["PREPROCESSING"]["NORMALIZE_HR_DN"]["STD"]
    return im_path,(thresh_lr,mean_lr,std_lr),(thresh_hr,mean_hr,std_hr)
    

