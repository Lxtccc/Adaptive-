import os
import sys
import glob
import numpy as np
import torch
import utils
import copy
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from thop import profile

from models.ResNet import ResNet20
from keep_rate_actor import PolicyGradient, learn

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='F://Code_implement//Dataset', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1000, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='num of coarsely training epochs')
parser.add_argument('--epochs_for_clip', type=int, default=50, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='Test5', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--valid_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--use_ThresNet', default=True)
parser.add_argument('--checkpoint', type=str, default='')
args = parser.parse_args()

args.save = os.path.join('Trained_experiments', args.save)
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10
action_choice = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')

  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  cudnn.enabled=True
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model = ResNet20(CLASS=CIFAR_CLASSES)
  flops, paras = profile(model, inputs=(torch.randn(1, 3, 32, 32), ))
  logging.info("model param size = %fMB, flops = %E", paras/1e6, flops)
  model = model.cuda()

  # difining ThresNet
  actor = PolicyGradient(action_choice=action_choice)
  actor = actor.cuda()

  ### Installing dataset
  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.valid_portion * num_train))

  train_ori_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    pin_memory=True, num_workers=2)
  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    pin_memory=True, num_workers=2)
  valid_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size,
    sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
    pin_memory=True, num_workers=2)
  test_queue = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size,
    pin_memory=True, num_workers=2)

  ### Coarsely train entire target network
  # model = Coarsely_training(model, train_ori_queue, test_queue, criterion)
  logging.info('~~~Training all parameters finished and start clip training.\n')
  logging.info('Rebuild the sub-graph')

  batch_soft_prob = []
  batch_actions = []
  batch_loss = []
  batch_counter = 0

  ### Start searching
  for epoch in range(args.epochs_for_clip):
    logging.info('================[ Clip Epoch %d ]================', epoch+1)
    ####################################### clip ########################################################
    logging.info('~~~Start sampling {}th'
                 ' sub-network and installing hyper-paras.'.format(epoch+1))
    ### pre-defining hyper-parameter
    if epoch <= 50:#40-exp01-02
      wn = True
    else:
      wn = False
    alpha = model.all_arch_parameters()

    soft_prob, actions, keep_rate, keeped_size = actor(alpha)

    batch_soft_prob.append(soft_prob)
    batch_actions.append(actions)
    batch_counter += 1

    ### Rebuild a total new subnet accroding to kept filters' number
    selected_index, selected_alpha, selected_lenth, keep_rate_list = \
      utils.sample_from_alpha_adaptive(alpha, keep_rate, with_noise=wn)
    logging.info('***[ keep_rate_list:{}, length:{} ]***'.format(keep_rate_list, selected_lenth))

    new_sub_model = ResNet20(CLASS=CIFAR_CLASSES, len_list= selected_lenth, subgraph = True)
    flops, paras = profile(new_sub_model, inputs=(torch.randn(1, 3, 32, 32),))
    logging.info("***[ sub_model flops = %E, paras = %E ]***", flops, paras)

    ### copy the parameters of sub-network from target network
    sub_model = utils.load_subgraph_from_model(new_sub_model, model, selected_index, selected_alpha)
    sub_model = sub_model.cuda()
    logging.info('~~~Rebuild finished and start training in sub-graph and Start training network parameters in sub-graph.')

    ####################################### training #####################################################
    model, losses = alternating_optimization(model, train_queue, valid_queue, test_queue, epoch, criterion)
    batch_loss.append(losses)

    ####################################### train ThresNet ##############################################
    optimizer_actor = torch.optim.Adam(
      actor.parameters(),
      lr=0.1
    )
    if batch_counter % 1 == 0:
      loss, model_size = learn(batch_soft_prob, batch_actions, batch_loss, sub_model_size=paras/1e6 * keeped_size,
                               opt=optimizer_actor)
      batch_soft_prob = []
      batch_actions = []
      batch_loss = []
      logging.info('~~~Train ThresNet finished!')

    ####################################### update parameter ############################################
    model = utils.update_model_from_subgraph(sub_model, model, selected_index)
    logging.info('~~~Training-and-Searching the No.{} sub-network finished\n'.format(epoch+1))

def Coarsely_training(model, train_data, test_data, criterion):
  optimizer_all = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_all, float(args.epochs))

  best_acc = 0
  sub_model_dic = []
  if args.pretrain:
    utils.load(model, args.checkpoint)
    test_acc, test_obj = test(test_data, model, criterion)
    logging.info('Load pretrained model with Acc=' + str(test_acc)[0:5] + '\n', )
  else:
    for epoch in range(args.epochs):
      lr = scheduler.get_lr()[0]

      ############ training ##################
      model, train_acc, train_obj = train(train_data, model, criterion, optimizer_all)
      ############ testing ##################
      if epoch == args.epochs-1:
        test_acc, test_obj = test(test_data, model, criterion)
        save_name = 'Coarsely_acc_' + str(test_acc)[0:5] + '.pt'
        torch.save(model, os.path.join(args.save, save_name))
      scheduler.step()
      logging.info('[ epoch:%d/%d lr:%f train_loss:%f train_acc:%f ]',
                   epoch, args.epochs, lr, train_obj, train_acc)
    logging.info('Coarsely_traned_with_acc=' + str(test_acc)[0:5] + '\n', )
  return model

def alternating_optimization(model, train_data, valid_data, test_data, epoch, criterion):
  if epoch <= 20:
    M = 40; lr = 0.05
  elif epoch > 20 and epoch <= 50:
    M = 30; lr = 0.01
  else:
    M = 10; lr = 0.005

  best_acc = 80
  sub_model_dic = []
  batch_loss = []
  optimizer_weight = torch.optim.SGD(
    model.parameters_without_alpha(),
    lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay)
  scheduler_weight = torch.optim.lr_scheduler.CosineWithRestarts(
    optimizer_weight, M)

  optimizer_alpha = torch.optim.SGD(
    model.parameters_only_alpha(),
    1e-3,
    momentum=args.momentum,
    weight_decay=1e-4)

  for i in range(M):
    model, train_acc, train_obj = train(train_data, model, criterion, optimizer_weight)
    batch_loss.append(train_obj)
    scheduler_weight.step()
    if i in list(range(0, M, M//5)):
      model, valid_acc, valid_obj, valid_loss = valid(valid_data, model, criterion, optimizer_alpha)
    test_acc, _ = test(test_data, model, criterion)
    if test_acc > best_acc:
      best_acc = test_acc
      save_name = 'Iter' + str(epoch + 1) + '_' + str(best_acc)[0:5] + '.pt'
      sub_model_dic = copy.deepcopy(model.state_dict())
    if i%10 == 0:
      logging.info('[ epoch:%d/%d train_loss:%f val_loss:%f train_acc:%f  test_acc:%f ]',
                   i, M, train_obj, valid_loss, train_acc, test_acc)
  if sub_model_dic:
    model.load_state_dict(sub_model_dic)
    torch.save(model, os.path.join(args.save, save_name))
  logging.info('***[ test_acc %f for iter %d ]***', best_acc, epoch + 1)

  return model, np.average(batch_loss)

def train_pruned_model(model, train_data, test_data, criterion, lr, N, epoch, best_acc, eph=0):
  optimizer_without_alpha = torch.optim.SGD(
    model.parameters_without_alpha(),
    lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay)

  scheduler_without_alpha = torch.optim.lr_scheduler.CosineWithRestarts(
    optimizer_without_alpha, N)
  sub_model_dic = []
  for i in range(N):
    model, train_acc, train_obj = train(train_data, model, criterion, optimizer_without_alpha)
    test_acc, _ = test(test_data, model, criterion)
    if test_acc > best_acc and epoch >= 0:
      best_acc = test_acc
      eph = i
      save_name = 'epoch_' + str(epoch) + '-acc_' + str(best_acc)[0:5] + '.pt'
      sub_model_dic = copy.deepcopy(model.state_dict())
    scheduler_without_alpha.step()
  if sub_model_dic:
    model.load_state_dict(sub_model_dic)
    logging.info('***[ test_acc %f at epoch%d/epoch%d ]***', best_acc, eph, N)
    utils.save(model, os.path.join(args.save, save_name))
  return model

def train_alpha(model, valid_data, test_data, criterion, lr):
  optimizer_only_alpha = torch.optim.SGD(
    model.parameters_only_alpha(),
    lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay
  )
  # alpha_list = []
  for i in range(5):
    model, valid_acc, valid_obj, val_loss = valid(valid_data, model, criterion, optimizer_only_alpha)
    # alpha_list.append(copy.deepcopy(new_sub_model.all_arch_parameters()[0]))
  logging.info('***[ valid_acc:%f, valid_loss:%f, alpha_penalty_item:%f ]***', valid_acc, valid_obj, val_loss)
  return model

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)
    loss.backward()
    optimizer.step()
    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    # if step % args.report_freq == 0:
    #   logging.info('---train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  return model, top1.avg, objs.avg

def valid(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(async=True)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    alpha = model.all_arch_parameters()
    val_loss = 1 * utils.cal_alpha_var_loss(alpha)

    loss-=val_loss

    loss.backward()
    # nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    # if step % args.report_freq == 0:
    #   logging.info('---valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return model, top1.avg, objs.avg, val_loss

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
      #   logging.info('---test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      return top1.avg, objs.avg

if __name__ == '__main__':
  main() 

