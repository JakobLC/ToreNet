import torch
from utils import (ToreNet,BucketDataset,create_argparser,number_of_parameters,train_loop)

def main(**kwargs):
    args = create_argparser()
    args = args.parse_args()
    args.__dict__.update(kwargs)

    model = ToreNet(args)
    model.to("cuda")

    dataset = BucketDataset(im_reshape=(args.im_size,args.im_size))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True)

    number_of_parameters(model)

    losses = {"train_loss": [],
          "vali_loss": [],
          "train_iou": [],
          "vali_iou": []}
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loop(model, optimizer, train_dataloader, val_dataloader, args, losses)

    print("args:", args)

if __name__=="__main__":
    main()