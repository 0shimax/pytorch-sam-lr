from functools import partial
from pathlib import Path
import argparse

import torch
import torch.nn.functional as F
from optimizer.sam import SAMSGD
from model.lr import LrNet
from data.loader import CvrDataset, loader
from feature.preprocess import load_data

    
    
def main(args):
    model = LrNet(
        in_dim=args.in_dim
        ,n_class=args.n_class)

    if Path(args.resume_model).exists():
        print("load model:", args.resume_model)
        model.load_state_dict(torch.load(args.resume_model))

    optimizer = SAMSGD(model.parameters(), lr=args.lr, momentum=0.9,  # momentum=0.0
                       weight_decay=0.0, nesterov=False, rho=0.05)

    train_data, train_targets, encoder = load_data(args.train_data_args)
    train_dataset = CvrDataset(train_data, train_targets)
    train_loader = loader(train_dataset, args.batch_size)
    train(args, model, optimizer, train_loader)
    
    
    args.test_data_args["encoder"] = encoder
    test_data, test_targets, _ = load_data(args.test_data_args)
    test_dataset = CvrDataset(test_data, test_targets)
    test_loader = loader(test_dataset, 1, shuffle=False)
    test(args, model, test_loader)
    

def train(args, model, optimizer, data_loader):
    def closure():
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target.squeeze(1))
        loss.backward()
        return loss
    
    model.train()
    for epoch in range(args.epochs):
        for i, (data, target) in enumerate(data_loader):
            model.zero_grad()

            # optimizer.zero_grad()
            # output = model(data)
            # loss = F.cross_entropy(output, target.squeeze(1))
            # loss.backward()

            loss = optimizer.step(closure)
            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epochs, i,
                  len(data_loader), loss.item()))

        # checkpointing
        torch.save(model.state_dict(),
                   '{}/fat_ffm_ckpt.pth'.format(args.out_dir))
    torch.save(model.state_dict(),
                '{}/fat_ffm_ckpt.pth'.format(args.out_dir))


def test(args, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            output = model(data)
            test_loss += torch.mean(F.cross_entropy(
                output, target.squeeze(1), size_average=False)).item()
            pred = output.argmax(1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(data_loader.dataset),
                  100. * correct / len(data_loader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default='./data/dummy', help='path to dataset')
    parser.add_argument('--n-class', type=int, default=2, help='number of class')
    parser.add_argument('--in-dim', type=int, default=9, help='feature dimension')
    parser.add_argument('--train-file-name', default='train_dummy.csv', help='path to train data')
    parser.add_argument('--test-file-name', default='test_dummy.csv', help='path to test data')
    parser.add_argument('--resume-model', default='./results/lr_sam.pth', help='path to trained model')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch-size', type=int, default=256, help='input batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--out-dir', default='./results', help='folder to output images and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    target = "converted"
    select_cols = ["converted", "hour", "is_app", "creative_ad_type", "domain", "device", "slot_size", "categories", "advertiser_company_id", "imp_banner_pos", "imp_tagid"]
    category_cols = ["hour", "creative_ad_type", "is_app", "domain", "device", "slot_size", "categories", "advertiser_company_id", "imp_banner_pos", "imp_tagid"]
    drop_cols = ["hour", "imp_tagid", "is_app"]
    train_args = {"path":Path(args.root_dir, args.train_file_name),
                  "target":target, 
                  "sort_key":"imp_time",
                  "select_cols":select_cols, 
                  "category_cols":drop_cols,
                  "drop_cols":drop_cols,
                  "is_app":"is_app",
                  "encoder":None}
    test_args = {"path":Path(args.root_dir, args.test_file_name),
                  "target":target, 
                  "sort_key":"imp_time",
                  "select_cols":select_cols, 
                  "category_cols":drop_cols,
                  "drop_cols":drop_cols,
                  "is_app":"is_app"} 
    args.train_data_args = train_args
    args.test_data_args = test_args
    main(args)