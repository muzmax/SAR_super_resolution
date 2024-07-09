import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import tqdm
from scipy import special
from scipy import signal
import numpy as np
import cv2
import pathlib
import os
import glob
import torch.nn as nn
from numpy import exp
from torch.autograd import Variable
import torch.nn.functional as F

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
    def __init__(self, transform, root_lr, root_hr = '', test=False):
        self.root = root_lr
        self.transform = transform
        self.test = test
        if self.test:
            # Predict Mode
            self.lr_names = glob.glob(os.path.join(root_lr,'*.npy'))
        else:
            # Training mode
            self.lr_names = glob.glob(os.path.join(root_lr,'*.npy'))
            self.hr_names = [os.path.join(root_hr, filepath.split('/')[-1]) for filepath in self.lr_names]           
            
    def __getitem__(self, idx):
        if self.test:
            im = np.load(self.lr_names[idx])
            assert len(im.shape) == 2
            im = im[:,:,np.newaxis]
            name = os.path.basename(self.lr_names[idx])
            name = os.path.splitext(name)[0]
            return self.transform(im).float(), name
        else:
            imLR = np.load(self.lr_names[idx])
            assert len(imLR.shape) == 2
            imLR = imLR[:,:,np.newaxis]
            
            imHR = np.load(self.hr_names[idx])    
            assert len(imHR.shape) == 2
            imHR = imHR[:,:,np.newaxis]
            
            return self.transform(imLR).float(), self.transform(imHR).float()
            
    def __len__(self):
        return len(self.lr_names)


def load_train(cfg, norm_param):
    # Init test dataset
    transform = transforms.Compose([ToTensor(), normalizeSAR(norm_param[0],norm_param[1],norm_param[2])])
    train_valid_dataset = SARdataset(transform, cfg["TRAIN"]["PATH_TO_LR"], cfg["TRAIN"]["PATH_TO_HR"])

    # Split it into training and validation sets
    nb_valid = int(cfg["TRAIN"]["VALID_RATIO"] * len(train_valid_dataset))
    nb_train = len(train_valid_dataset) - nb_valid
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(
    train_valid_dataset,
    [nb_train, nb_valid],
    )
    # Build Loader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"])
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=cfg["TRAIN"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"])        
    return train_loader, valid_loader


def train_one_epoch(model, loader, f_loss, optimizer, device):
    """Train the model for one epoch

    Args:
        model (torch.nn.module): the architecture of the network
        loader (torch.utils.data.DataLoader): pytorch loader containing the data
        f_loss (torch.nn.module): Cross_entropy loss for classification
        optimizer (torch.optim.Optimzer object): adam optimizer
        device (torch.device): cuda

    Return:
        tot_loss : computed loss over one epoch
    """

    # scaler = torch.cuda.amp.GradScaler()
    model.train()

    n_samples = 0
    tot_loss = 0.0

    for low, high in tqdm.tqdm(loader):
        low, high = low.to(device), high.to(device)
        # with torch.cuda.amp.autocast():
        # Compute the forward pass through the network up to the loss
        outputs = model(low)
        loss = f_loss(outputs, high)
        tot_loss += low.shape[0] * loss.item()

        n_samples += low.shape[0]

        # Backward and optimize
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return tot_loss / n_samples


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window

def calculate_psnr(img1, img2, scale=256, border=0):
    """Function to computer peak to signal ratio

    Args:
        img1 ([type]): [description]
        img2 ([type]): [description]
        border (int, optional): [description]. Defaults to 0.

    Raises:
        ValueError: [description]

    Returns
        [type]: [description]
    """
    # img1 and img2 have range [-60, 0]

    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    h, w = img1.shape[2:]
    img1 = img1[:, :, border : (h - border), border : (w - border)]
    img2 = img2[:, :, border : (h - border), border : (w - border)]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2, axis=(2, 3))
    mse[mse == 0] = float("inf")

    return np.mean(10 * np.log10(scale**2 / mse))  # <=> 1/B * sum_i



def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return -_ssim(img1, img2, window, self.window_size, channel, self.size_average)


def valid_one_epoch(model, loader, f_loss, device):
    """Train the model for one epoch

    Args:
        model (torch.nn.module): the architecture of the network
        loader (torch.utils.data.DataLoader): pytorch loader containing the data
        f_loss (torch.nn.module): Cross_entropy loss for classification
        device (torch.device): cuda

    Return:
        tot_loss : computed loss over one epoch
    """
    ssim = SSIMLoss(size_average=True)
    with torch.no_grad():

        model.eval()

        n_samples = 0
        tot_loss = 0.0
        tot_l1loss = 0.0
        tot_l2loss = 0.0
        tot_ssim = 0.0
        avg_psnr = 0
        tot_huberloss = 0.0

        for low, high in tqdm.tqdm(loader):
            low, high = low.to(device), high.to(device)

            # with torch.cuda.amp.autocast():
            # Compute the forward pass through the network up to the loss
            outputs = model(low)

            batch_size = low.shape[0]

            # WARN: if using reduction = "mean", the avg
            # is computed over batch_size * Height * Width
            l1_loss = torch.nn.functional.l1_loss(outputs, high, reduction="mean")
            tot_l1loss += batch_size * l1_loss.item()
            l2_loss = torch.nn.functional.mse_loss(outputs, high, reduction="mean")
            tot_l2loss += batch_size * l2_loss.item()
            huber_loss = torch.nn.functional.huber_loss(outputs, high,  reduction="mean")
            tot_huberloss +=  batch_size * huber_loss.item()
            ssim_loss = ssim(outputs, high)
            tot_ssim += batch_size * ssim_loss.item()

            n_samples += batch_size
            tot_loss += batch_size * f_loss(outputs, high).item()

            # We need to denormalize the PSNR to correctly average
            psnr = batch_size * calculate_psnr(
                outputs.cpu().numpy(), high.cpu().numpy()
            )
            avg_psnr += psnr

        return (
            tot_loss / n_samples,
            avg_psnr / n_samples,
            low.cpu().numpy(),
            outputs.cpu().numpy(),
            high.cpu().numpy(),
            tot_l1loss / n_samples,
            tot_l2loss / n_samples,
            -tot_ssim / n_samples,
            tot_huberloss / n_samples,
        )


def load_test(cfg,norm_param):
    # Init test dataset
    transform = transforms.Compose([ToTensor(),normalizeSAR(norm_param[0],norm_param[1],norm_param[2])])
    test_data = SARdataset(transform, cfg["INFERENCE"]["PATH_TO_SLC"], test = True)
    # Build Loader
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data,
        batch_size=cfg["INFERENCE"]["BATCH_SIZE"],
        shuffle=False,
        num_workers=cfg["DATASET"]["NUM_THREADS"])
    return test_loader


def get_loss(cfg):
    """This function returns the loss from the config

    Args:
        cfg (dic): config file

    Returns:
        loss: loss
    """
    if cfg["TRAIN"]["LOSS"]["NAME"] == "SSIM":
        return SSIMLoss()
    elif cfg["TRAIN"]["LOSS"]["NAME"] == "l1":
        return nn.L1Loss()
    elif cfg["TRAIN"]["LOSS"]["NAME"] == "l2":
        return nn.MSELoss()
    elif cfg["TRAIN"]["LOSS"]["NAME"] == "l2sum":
        return nn.MSELoss(reduction="sum")
    elif cfg["TRAIN"]["LOSS"]["NAME"] == "huber":
        return nn.HuberLoss()
    else:
        raise NotImplementedError(f"Loss type [{cfg['TRAIN']['LOSS']}] is not found.")


def get_optimizer(cfg, params):
    """This function returns the correct optimizer

    Args:
        cfg (dic): config

    Returns:
        torch.optimizer: train optimizer
    """
    if cfg["TRAIN"]["OPTIMIZER"]["NAME"] == "Adam":
        return torch.optim.Adam(
            params,
            lr=cfg["TRAIN"]["OPTIMIZER"]["ADAM"]["LR_INITIAL"],
            weight_decay=cfg["TRAIN"]["OPTIMIZER"]["ADAM"]["WEIGHT_DECAY"],
        )
    else:
        raise NotImplementedError(
            "Optimizer type [{:s}] is not found.".format(
                cfg["TRAIN"]["OPTIMIZER"]["NAME"]
            )
        )

def get_scheduler(cfg, optimizer):
    """This function returns the correct learning rate scheduler

    Args:
        cfg (dic): config

    Returns:
        torch.optim.lr_scheduler: learning rate scheduler
    """
    if not "SCHEDULER" in cfg["TRAIN"]:
        return None
    if cfg["TRAIN"]["SCHEDULER"]["NAME"] == "ReduceOnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=cfg["TRAIN"]["SCHEDULER"]["ReduceOnPlateau"]["PATIENCE"],
            threshold=cfg["TRAIN"]["SCHEDULER"]["ReduceOnPlateau"]["THRESH"],
        )
    else:
        raise NotImplementedError(
            "Scheduler type [{:s}] is not found.".format(
                cfg["TRAIN"]["SCHEDULER"]["NAME"]
            )
        )


class ModelCheckpoint:

    """Define the model checkpoint class"""

    def __init__(self, dir_path, model, epochs, checkpoint_step):
        self.min_loss = None
        self.dir_path = dir_path
        self.best_model_filepath = os.path.join(self.dir_path, "best_model.pth")
        self.model = model
        self.epochs = epochs
        self.checkpoint_step = checkpoint_step

    def update(self, loss, epoch):
        """Update the model if the we get a smaller lost

        Args:
            loss (float): Loss over one epoch
        """

        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.best_model_filepath)
            self.min_loss = loss

        if epoch in np.arange(
            self.checkpoint_step - 1, self.epochs, self.checkpoint_step
        ):

            print(f"Saving model at Epoch {epoch}")

            filename = "epoch_" + str(epoch) + "_model.pth"

            filepath = os.path.join(self.dir_path, filename)
            torch.save(self.model.state_dict(), filepath)

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



