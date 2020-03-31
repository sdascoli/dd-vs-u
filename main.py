import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy
from collections import defaultdict
from utils import rot
from utils import get_data, hinge_regression, hinge_classification
from model import FullyConnected
import argparse

def train(model, tr_data, crit, opt, epochs):
        
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for (x,y) in tr_data:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            epoch_loss += loss.item()/len(tr_data)
            opt.step()
        losses.append(epoch_loss)
    return epoch_loss

def test(model, te_data, crit, task):
    with torch.no_grad():
        for (x,y) in te_data:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_loss = crit(out, y).item()
            if task=='classification':
                preds = out.max(1)[1]
                test_acc = preds.eq(y).sum().item().float()/len(y)
            else:
                test_acc = 0
            break
    return test_loss, test_acc
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--no_cuda', default=False, type=bool)

    parser.add_argument('--task', default='classification', type=str)
    parser.add_argument('--loss_type', default='default', type=str)
    parser.add_argument('--epsilon', default=.1, type=float)

    parser.add_argument('--depth', default=1, type=int)
    parser.add_argument('--teacher_width', default=100, type=int)
    parser.add_argument('--teacher_depth', default=2, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--min_width', default=1, type=int)
    parser.add_argument('--max_width', default=100, type=int)
    parser.add_argument('--num_widths', default=20, type=int)
    
    parser.add_argument('--n', default=100, type=int)
    parser.add_argument('--n_test', default=10000, type=int)
    parser.add_argument('--noise', default=0.1, type=float)
    parser.add_argument('--test_noise', default=True, type=bool)
    parser.add_argument('--d', default=2, type=int)
    parser.add_argument('--var', default=.5, type=int)
    parser.add_argument('--n_classes', default=1, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--mom', default=0.9, type=float)
    parser.add_argument('--bs', default=1000, type=int)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.no_cuda: device='cpu'

    if args.task=='classification':
        teacher = None
        if args.loss_type == 'linear_hinge':
            crit = lambda x,y : hinge_classification(x,y, epsilon=args.epsilon, type='linear')
        elif args.loss_type == 'quadratic_hinge':
            crit = lambda x,y : hinge_classification(x,y, epsilon=args.epsilon, type='quadratic')
        elif args.loss_type == 'nll':
            crit = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError
    elif args.task=='regression':
        with torch.no_grad():
            teacher = FullyConnected(width=args.teacher_width, n_layers=args.teacher_depth, in_dim=args.d, out_dim=args.n_classes).to(device)
        if args.loss_type == 'linear_hinge':
            crit = lambda x,y : hinge_regression(x,y, epsilon=args.epsilon, type='linear')
        elif args.loss_type == 'quadratic_hinge':
            crit = lambda x,y : hinge_regression(x,y, epsilon=args.epsilon, type='quadratic')
        elif args.loss_type == 'mse':
            crit = nn.MSELoss()
        else:
            raise NotImplementedError
    else:
        raise 
            
    bs = min(args.bs, args.n)
    n_batches = int(args.n/bs)
    tr_data = get_data(args.task, n_batches, bs, args.d, args.noise, n_classes=args.n_classes,  var=args.var, teacher=teacher)
    test_noise = args.noise if args.test_noise else 0
    te_data = get_data(args.task, 1, args.n_test, args.d, test_noise, n_classes=args.n_classes, var=args.var, teacher=teacher)
        
    widths = np.unique(np.logspace(np.log10(args.min_width), np.log10(args.max_width), args.num_widths).astype(int))    
    train_losses = []
    test_losses  = []
    test_accs     = []


    for width in widths:
        student = FullyConnected(width=width, n_layers=args.depth, in_dim=args.d, out_dim=args.n_classes).to(device)
        opt = torch.optim.SGD(student.parameters(), lr=args.lr, momentum=args.mom)
        train_loss = train(student, tr_data, crit, opt, args.epochs)
        test_loss, test_acc = test(student, te_data, crit, args.task)
        print(test_loss)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    dic = {'args':args, 'widths':widths, 'train_loss':train_losses, 'test_loss':test_losses, 'test_acc':test_accs}
    torch.save(dic, args.name+'.pyT')
