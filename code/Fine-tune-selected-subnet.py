import torch
import torch.nn as nn
import os
import sys
import numpy as np
import utils
import logging
import argparse
from thop import profile
import time
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
from torch.autograd import Variable

from models.ResNet import ResNet20

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
CIFAR_CLASSES = 10

parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--data', type=str, default='../../../../data', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-6, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=800, help='num of training epochs')
parser.add_argument('--epochs_for_clip', type=int, default=300, help='num of training epochs')
parser.add_argument('--save', type=str, default='ft', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
args = parser.parse_args()



def main():
    file_name = 'acc-62.46_epoch_79.pt'
    file_name = os.path.join(os.path.dirname(os.getcwd()), file_name)
    pt_file = torch.load(file_name)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)


    cudnn.benchmark = True
    cudnn.enabled = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.save = 'Finetuned_model6246'
    utils.create_exp_dir(args.save)

    channel_size = []
    for k in pt_file:
        if 'alpha' in k:
            channel_size.append((pt_file[k].shape[0]))

    purned_model = ResNet50(CLASS=CIFAR_CLASSES, len_list = channel_size, subgraph=True)
    purned_model.load_state_dict(pt_file)

    flops, paras = profile(purned_model, inputs=(torch.randn(1, 3, 32, 32),))
    purned_model = purned_model.cuda()
    print("sub model param size = ", utils.count_parameters_in_MB(purned_model), "MB")
    print("model flops = %E", flops)
    # stat(purned_model, (3,32,32))
    # utils.print_model_parm_flops(purned_model, input=torch.randn(1,3,32,32))
    # flops, params = profile(purned_model, input_size=(1, 3, 32, 32))

    optimizer_all = torch.optim.SGD(
        purned_model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR100(root=args.data, train=True, download=False, transform=train_transform)
    test_data = dset.CIFAR100(root=args.data, train=False, download=False, transform=valid_transform)


    num_train = len(train_data)
    indices = list(range(num_train))

    train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      pin_memory=True, num_workers=2)

    test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size,
      pin_memory=True, num_workers=2)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_all, float(args.epochs))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    best_acc = 60

    start = time.time()

    for epoch in range(args.epochs):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        print('epoch:', epoch,' lr:',  lr)
        ############ training ##################
        _, train_acc, train_obj = train(train_queue, purned_model, criterion, optimizer_all)
        print('train_acc :', train_acc)
        ############ testing ##################
        test_acc, test_obj = test(test_queue, purned_model, criterion)
        print('test_acc  :', test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            utils.save(purned_model, os.path.join(args.save, 'finetune_all_weights.pt'))
            end = time.time()
        print('\n')
    os.rename(os.path.join(args.save, 'finetune_all_weights.pt'), os.path.join(args.save, str(best_acc[0:5]) + '.pt'))
    print('best_acc  :', best_acc)
    print('ft_time  :', end-start)

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    # get a random minibatch from the search queue with replacement
    # input_search, target_search = next(iter(valid_queue))
    # input_search = Variable(input_search, requires_grad=False).cuda()
    # target_search = Variable(target_search, requires_grad=False).cuda(async=True)

    # architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return model, top1.avg, objs.avg

def test(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(test_queue):
    with torch.no_grad():
      input = Variable(input).cuda()
      target = Variable(target).cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    # if step % args.report_freq == 0:
    #   logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

if __name__ == '__main__':
  main()
