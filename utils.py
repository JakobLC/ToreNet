import torch
from PIL import Image
from pathlib import Path
import numpy as np
from scipy import ndimage as nd
import tqdm
import argparse
from diffusers import UNet2DConditionModel

def train_loop(model, optimizer, train_dataloader, val_dataloader, args, losses):
    model.train()
    vali_iou = [0]
    vali_loss = 0
    with tqdm.tqdm(range(args.num_epochs*len(train_dataloader))) as pbar:
        for epoch in range(args.num_epochs):
            for i, batch in enumerate(train_dataloader):
                optimizer.zero_grad()
                features, image, ground_truth = batch
                features, image, ground_truth = (features.to(args.device), 
                                        image.to(args.device), 
                                        ground_truth.to(args.device))
                pred = model(image, features)
                loss = torch.nn.functional.cross_entropy(pred, ground_truth.squeeze(1))
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
                train_iou = get_class_jaccard_index(pred,ground_truth,args.num_classes)
                losses["train_loss"].append(train_loss)
                losses["train_iou"].append(train_iou)
                pbar.set_description(f"train loss: {train_loss:.3f}, train iou: {sum(train_iou)/len(train_iou):.3f}, "
                                    +f"vali loss: {vali_loss:.3f}, vali iou: {sum(vali_iou)/len(vali_iou):.3f}")
                pbar.update(1)
            with torch.no_grad():
                model.eval()
                vali_loss = 0
                vali_iou = [0 for _ in range(args.num_classes)]
                for i, batch in enumerate(val_dataloader):
                    features, image, ground_truth = batch
                    features, image, ground_truth = (features.to(args.device), 
                                                    image.to(args.device), 
                                                    ground_truth.to(args.device))
                    pred = model(image, features)
                    loss = torch.nn.functional.cross_entropy(pred, ground_truth.squeeze(1))
                    vali_loss += loss.item()/len(val_dataloader)
                    vali_iou = [x+y/len(val_dataloader) for x,y in zip(vali_iou,get_class_jaccard_index(pred,ground_truth,args.num_classes))] 
                losses["vali_loss"].append(vali_loss)
                losses["vali_iou"].append(vali_iou)
                model.train()

def get_class_jaccard_index(pred,gt,num_classes):
    assert gt.shape[1]==1
    if not pred.shape[1]==1:
        pred = pred.clone().argmax(1).unsqueeze(1)
    jaccard_index = []
    for i in range(num_classes):
        pred_i = pred==i
        gt_i = gt==i
        intersection = torch.logical_and(pred_i,gt_i).sum().item()
        union = torch.logical_or(pred_i,gt_i).sum().item()
        jaccard_index.append(intersection/union)
    return jaccard_index

def number_of_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model has {num_params} trainable parameters")

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if isinstance(v, list):
            if isinstance(v[0], int):
                v_type = lambda x: [int(xi) for xi in x.split(",")] if isinstance(x, str) else list(x)
            else:
                v_type = lambda x: x.split(",") if isinstance(x, str) else list(x)
        elif isinstance(v, bool):
            v_type = str2bool
        elif v is None:
            v_type = str
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
    
def create_argparser():
    defaults = get_default_args()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults.__dict__)
    return parser

def get_dummy_batch(im_size=256, batch_size=32, feature_dim1=76, feature_dim2=32,num_classes=3,
                    noisyness_image=0.95,
                    noisy_features=0.5):
    
    ground_truth = np.random.rand(batch_size, num_classes, im_size, im_size)
    #blur ground truth and make into class pixel map:
    sigma = im_size/10
    ground_truth = nd.gaussian_filter(ground_truth, sigma=(0,0,sigma,sigma))
    ground_truth[:,0] += ground_truth.std()*1.0
    ground_truth = np.argmax(ground_truth, axis=1)[:,None]
    image = np.random.rand(batch_size, 1, im_size, im_size)
    image = noisyness_image*image + (1-noisyness_image)*ground_truth/num_classes
    features = np.random.rand(batch_size, 1, feature_dim1, feature_dim2)
    ground_truth_resized = torch.nn.functional.interpolate(torch.tensor(ground_truth).float(), 
                                                           size=(feature_dim1, feature_dim2), 
                                                           mode="nearest").numpy()
    features = noisy_features*features + (1-noisy_features)*ground_truth_resized/num_classes
    return features, image, ground_truth

class BucketDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir="./data", im_reshape=(64,64)):
        self.root_dir = root_dir
        self.im_reshape = im_reshape
        self.folder_names = ["features", "images", "ground_truth"]
        self.filenames = []
        for path in list((Path(root_dir)/ self.folder_names[0]).glob("*")):
            self.filenames.append(path.name)
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        for folder_name in self.folder_names:
            path = Path(self.root_dir)/folder_name/filename
            if folder_name == self.folder_names[0]:
                #load features
                features = torch.from_numpy(np.array(Image.open(path))/255).float().unsqueeze(0)
            elif folder_name == self.folder_names[1]:
                #load image
                image = torch.from_numpy(np.array(Image.open(path))/255).float().unsqueeze(0).unsqueeze(0)
            elif folder_name == self.folder_names[2]:
                #load ground truth
                ground_truth = torch.from_numpy(np.array(Image.open(path))).float().unsqueeze(0).unsqueeze(0)
        image = torch.nn.functional.interpolate(image, size=self.im_reshape, mode="area").squeeze(0)
        ground_truth = torch.nn.functional.interpolate(ground_truth, size=self.im_reshape, mode="area").long().squeeze(0)
        sample = [features,image,ground_truth]
        return sample
    
def convert_unet_to_downnet(unet):
    unet.up_blocks = torch.nn.ModuleList()
    unet.conv_norm_out = None
    unet.conv_out = torch.nn.Identity()
    return unet

def get_default_args():
    args = argparse.Namespace()
    #training args
    args.batch_size = 4
    args.lr = 0.001
    args.num_epochs = 10
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model args
    args.im_size=64
    args.feature_dim1=76
    args.feature_dim2=32
    args.block_out_channels = [32,32,64,64,128] #how large the matrices are in the unet
    args.num_classes = 3 # number of blobs+ background class

    # One of ["concat","embed","simple_x_attn","x_attn"] where each is more complex than the one before. Not sure what is best
    args.conditioning_mode = "x_attn"

    #args you probably dont want to change
    args.encoder_hid_dim = 64
    args.cross_attention_dim = 64
    args.down_block_types = ["DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"]
    args.up_block_types = ["UpBlock2D", "UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"]
    args.mid_block_type = "UNetMidBlock2DCrossAttn"
    
    return args

def modify_block_names(block_names,conditioning_mode):
    if conditioning_mode in ["concat","embed"]:
        block_names = [x.replace("SimpleCrossAttn","") for x in block_names]
        block_names = [x.replace("CrossAttn","") for x in block_names]
    elif conditioning_mode == "simple_x_attn":
        block_names = [x.replace("CrossAttn","SimpleCrossAttn") for x in block_names]
    else:
        assert conditioning_mode == "x_attn"
    return block_names

def is_power_of_2(num):
    power = np.log2(num)
    is_power = np.isclose(power,np.round(power))
    return 2**np.ceil(power).astype(int),is_power

class ToreNet(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args

        down_block_types = args.down_block_types
        while len(args.block_out_channels) != len(down_block_types):
            down_block_types = [args.down_block_types[0]] + args.down_block_types
        up_block_types = args.up_block_types
        while len(args.block_out_channels) != len(up_block_types):
            up_block_types = [args.up_block_types[0]] + args.up_block_types
        if args.conditioning_mode in ["concat","embed"]:
            mid_block_type = None
        elif args.conditioning_mode == "simple_x_attn":
            mid_block_type = args.mid_block_type.replace("CrossAttn","SimpleCrossAttn")
        elif args.conditioning_mode == "x_attn":
            mid_block_type = args.mid_block_type
        down_block_types = modify_block_names(down_block_types,args.conditioning_mode)
        up_block_types = modify_block_names(up_block_types,args.conditioning_mode)
        self.image_unet = UNet2DConditionModel(block_out_channels=args.block_out_channels,
                                         encoder_hid_dim=args.encoder_hid_dim if args.conditioning_mode != "concat" else None,
                                         cross_attention_dim=args.cross_attention_dim,
                                         down_block_types=down_block_types,
                                         up_block_types=up_block_types,
                                         mid_block_type=mid_block_type,
                                         in_channels=2 if args.conditioning_mode == "concat" else 1, 
                                         out_channels=args.num_classes,
                                         addition_embed_type="text" if args.conditioning_mode == "embed" else None)
        assert args.im_size % 2**(len(args.block_out_channels)-1) == 0, "image size must be divisible by 2**num_down_blocks"
        if args.conditioning_mode in ["embed","simple_x_attn","x_attn"]:
            down_block_types = modify_block_names(down_block_types,"embed")
            up_block_types = modify_block_names(up_block_types,"embed")
            self.feature_downnet = UNet2DConditionModel(block_out_channels=args.block_out_channels[:-1]+[args.encoder_hid_dim],
                                         down_block_types=down_block_types,
                                         up_block_types=up_block_types,
                                         mid_block_type=None,
                                         in_channels=2 if args.conditioning_mode == "concat" else 1, 
                                         out_channels=args.num_classes)
            self.feature_unet = convert_unet_to_downnet(self.feature_downnet)
            
            #add so we can accept strange feature sizes
            power_ceil1,is_power1 = is_power_of_2(args.feature_dim1)
            self.feature_dim1_linear = torch.nn.Identity() if is_power1 else torch.nn.Linear(args.feature_dim1, power_ceil1)
            power_ceil2,is_power2 = is_power_of_2(args.feature_dim2)
            self.feature_dim2_linear = torch.nn.Identity() if is_power2 else torch.nn.Linear(args.feature_dim2, power_ceil2)
            out_dim = np.round(args.encoder_hid_dim*power_ceil1*power_ceil2/(2**(len(args.block_out_channels)-1))**2).astype(int)
            if out_dim>4096:
                print("Warning: out_dim is very large, you might want to change the feature_dim1 and feature_dim2 args, or add more blocks")
            self.feature_fc = torch.nn.Sequential(*[torch.nn.Linear(out_dim, 2*args.encoder_hid_dim),
                                                    torch.nn.SiLU(),
                                                    torch.nn.Linear(2*args.encoder_hid_dim, args.encoder_hid_dim),
                                                    torch.nn.SiLU(),
                                                    torch.nn.Linear(args.encoder_hid_dim, args.encoder_hid_dim)])
    def forward(self, image, features):
        dummy_timesteps = torch.tensor(0).long().cuda() #added simply because this library is made for diffusion models and we dont want to rewrite their code
        if self.args.conditioning_mode == "concat":
            features = torch.nn.functional.interpolate(features, size=(image.shape[2],image.shape[3]), mode="area")
            x = torch.concat([image, features], dim=1)
            x = self.image_unet(x, dummy_timesteps, encoder_hidden_states=None)
        elif self.args.conditioning_mode in ["embed","simple_x_attn","x_attn"]:
            features = self.feature_dim2_linear(features)
            features = self.feature_dim1_linear(features.permute(0,1,3,2)).permute(0,1,3,2)
            features = self.feature_downnet(features, dummy_timesteps, encoder_hidden_states=None)["sample"]
            features = self.feature_fc(features.flatten(1))[:,None,:]
            encoder_hidden_states = features
            x = self.image_unet(image, dummy_timesteps, encoder_hidden_states=encoder_hidden_states)
        else:
            raise NotImplementedError(self.args.conditioning_mode)
        return x["sample"]