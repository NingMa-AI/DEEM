'''
Released by Ning Ma for DEEM model

This repository is based on SHOT https://github.com/tim-learn/SHOT 
and MME https://github.com/VisionLearningGroup/SSDA_MME


'''

# from utils import setup_seed
import argparse
import os
import os.path as osp
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from models.SSDA_basenet import *
from models.SSDA_resnet import *
from copy import deepcopy
import contextlib
from data_pro.return_dataset import return_dataset,return_dataloader_by_UPS
import scipy
import scipy.stats
from itertools import cycle

def train_source(args):
    # if os.path.exists(osp.join(args.output_dir, "source_C_val.pt")):
    #   print("train_file exist,",args.output_dir)
    #   return 0
    source_loader,source_val_loader, _, _, target_loader_val, \
    target_loader_test, class_list = return_dataset(args)
    netF,netC,_=get_model(args)
    netF=netF.cuda()
    netC=netC.cuda()
    param_group = []
    learning_rate = args.lr
    for k, v in netF.features.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': learning_rate}]

    for k, v in netF.bottle_neck.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': learning_rate*10}]

    for k, v in netC.named_parameters():
        v.requires_grad = True
        param_group += [{'params': v, 'lr': learning_rate*10}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                factor=0.5, patience=50,
                                                                verbose=True, min_lr=1e-6)
    acc_init = 0
    for epoch in (range(args.max_epoch)):
        netF.train()
        netC.train()
        total_losses,recon_losses,classifier_losses=[],[],[]
        iter_source = iter(source_loader)
        for _, (inputs_source, labels_source) in tqdm(enumerate(iter_source), leave=False):
            if inputs_source.size(0) == 1:
                continue
           
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            embeddings=netF(inputs_source)

            outputs_source = netC(embeddings)

            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1)(outputs_source,
                                                                                             labels_source,T=1)
            total_loss=classifier_loss
            total_losses.append(total_loss.item())
            classifier_losses.append(classifier_loss.item())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        netF.eval()
        netC.eval()

        scheduler.step(np.mean(total_losses))
        acc_s_tr, _ = cal_acc(source_loader, netF, netC)
        acc_s_te, _ = cal_acc(source_val_loader, netF, netC)
        log_str = 'train_source , Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%, total_loss: {:.6f}, classify loss: {:.6f},'.format(args.s+"2"+args.t, epoch + 1, args.max_epoch,
                                                                             acc_s_tr * 100, acc_s_te * 100,np.mean(total_losses),np.mean(classifier_losses))
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str + '\n')

        if acc_s_te >= acc_init:
            acc_init = acc_s_te
            best_netF = netF.state_dict()
            best_netC = netC.state_dict()
            torch.save(best_netF, osp.join(args.output_dir, "source_F_val.pt"))
            torch.save(best_netC, osp.join(args.output_dir, "source_C_val.pt"))

    return netF, netC

def test_target(args):
    _, _,_, _, _, target_loader_test, class_list = return_dataset(args)
    netF, netC, _ = get_model(args)

    args.modelpath = args.output_dir + '/source_F_val.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C_val.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netC=netC.cuda()
    netF=netF.cuda()
    netF.eval()
    netC.eval()

    acc, _ = cal_acc(target_loader_test, netF, netC)
    log_str = 'test_target Task: {}, Accuracy = {:.2f}%'.format(args.s+"2"+args.t, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

def train_target(args):
    source_loader, source_val_loader, target_loader, target_loader_unl, \
    target_loader_val, target_loader_test, class_list=return_dataset(args)
    len_target_loader=len(target_loader)

    netF, netC, netD = get_model(args)

    args.modelpath = args.output_dir + '/source_F_val.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir + '/source_C_val.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF = netF.cuda()
    netC = netC.cuda()
    # netD = netD.cuda()

    param_group = []

    for k, v in netF.features.named_parameters():
        v.requires_grad=True
        param_group += [{'params': v, 'lr': args.lr}]

    for k, v in netF.bottle_neck.named_parameters():
        v.requires_grad=True
        param_group += [{'params': v, 'lr': args.lr*10}]

    for k, v in netC.named_parameters():
        v.requires_grad = False
        # param_group += [{'params': v, 'lr': args.lr}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=30,
                                                           verbose=True, min_lr=1e-6)
    vat_loss=VATLoss()

    max_pred_acc=-1
    best_test_acc = -1
    best_F,bestC,bestD=None,None,None
    first_epoch_acc=-1
    psudo_acc=-1
    for epoch in (range(args.max_epoch)):
        if args.max_num_per_class>0:
            active_target_loader, active_target_loader_unl, psudo_acc= return_dataloader_by_UPS(args,netF,netC,target_loader_unl)
            target_loader=active_target_loader
            target_loader_unl=active_target_loader_unl

        total_losses, l_classifier_losses,unl_classifier_losses, entropy_losses,labeled_entropy_losses,div_losses,m_losses= [], [],[],[],[],[],[]
        netF.eval()
        netC.eval()
        if epoch %1==0:
            mem_label=label_propagation(target_loader_test,target_loader,netF,netC,args)
            mem_label = torch.from_numpy(mem_label)
        netF.train()
        netC.train()
        for step, ((labeled_target,label),(unlabeled_target, labels_target ,idx)) in tqdm(enumerate(zip(cycle(target_loader), target_loader_unl)), leave=False):
            if unlabeled_target.size(0) == 1:
                continue
            len_label=labeled_target.size(0)
            inputs=torch.cat((labeled_target,unlabeled_target),dim=0).cuda()
            target_features = netF(inputs)

            target_out = netC(target_features,reverse=False)

            labels_target=labels_target.cuda()

            unlabeled_target_out=target_out[len_label:]
            labeled_target_pred=target_out[0:len_label]

            classifier_loss=0
            im_loss=0
            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0)(labeled_target_pred, label)
            l_classifier_losses.append(classifier_loss.item())
            pred_label=mem_label[idx]

            unl_loss = args.unl_w*CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0)(unlabeled_target_out, pred_label)
            classifier_loss+=unl_loss
            unl_classifier_losses.append(unl_loss.item())

            m_loss = vat_loss(netF,netC,inputs)
            im_loss += args.vat_w*m_loss
            m_losses.append(args.vat_w*m_loss.item())

            softmax_out = nn.Softmax(dim=1)(unlabeled_target_out)
            un_labeled_entropy = torch.mean(Entropy(softmax_out))
            im_loss+=args.unlent*un_labeled_entropy
            entropy_losses.append(un_labeled_entropy.item())


            msoftmax = softmax_out.mean(dim=0)
            tmp = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            div_losses.append(tmp.item())
            im_loss -=args.div_w*tmp

            total_loss = args.im * im_loss + args.par * classifier_loss
            total_losses.append(total_loss.item())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


        netF.eval()
        netC.eval()
        acc, _ = cal_acc(target_loader_test, netF, netC)
        acc_val, _ = cal_acc(target_loader_val, netF, netC)

        scheduler.step(np.mean(total_losses))
        log_str = 'tra_tgt: {}, I:{}/{}; test_acc = {:.2f}% ,acc_val = {:.2f}% ,total_l: {:.2f}, L_cls_l: {:.2f}, UNL_cls_l: {:.2f},' \
                  'ent_l: {:.2f},  div_loss: {:.2f},  vat_l: {:.2f} pseudo_label_ACC{:.2f}'.format(
            args.s+"2"+args.t, epoch + 1, args.max_epoch, acc * 100,acc_val*100,
            np.mean(total_losses),np.mean(l_classifier_losses),np.mean(unl_classifier_losses),np.mean(entropy_losses),
            np.mean(div_losses),np.mean(m_losses),psudo_acc)
        args.out_file.write(log_str + '\n')
        args.out_file.flush()
        print(log_str + '\n')
        if max_pred_acc<acc_val:
            best_F=deepcopy(netF)
            bestC=deepcopy(netC)
            max_pred_acc=acc_val

        if best_test_acc<acc:
            best_test_acc=acc

    
    acc, _ = cal_acc(target_loader_test, best_F, bestC)
    log_str="test_acc: {:.4f} ".format(acc)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return netF, netC

def Entropy(input_):
    # bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, netF,netC, x):
        # print(torch.mean(x))
        with torch.no_grad():
            pred = F.softmax(netC(netF(x)), dim=1)
            # softmax_out = nn.Softmax(dim=1)(pred)
            # entropy = Entropy(softmax_out)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        # print("d",torch.mean(d))
        with _disable_tracking_bn_stats(netF):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = netC(netF(x + self.xi * d))
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                netF.zero_grad()
                netC.zero_grad()

            # calc LDS
            # r_adv = d * self.eps
            # print("d, entropy", d.shape, torch.mean(entropy))
            # entropy=entropy.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 3, 224, 224)
            # entropy=torch.sigmoid(entropy)

            # r_adv = d * self.eps*(10/torch.exp(entropy))
            r_adv = d * self.eps
            # r_adv = d * self.eps*entropy
            pred_hat = netC(netF((x + r_adv)))
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

def inferance(loader,netF,netC,args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas = netF(inputs)
            outputs = netC(feas)
            # yield (feas,outputs,labels)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    return all_fea,all_output,all_label

def label_propagation(loader, labeled_loader, netF, netC, args):
    all_fea, all_output, all_label = inferance(loader, netF, netC, args)
    K = all_output.size(1)
    all_fea_labeled, all_output_labeled, all_label_labeled = inferance(labeled_loader, netF, netC, args)
    max_iter = 40
    alpha = 0.90

    if args.dataset =="multi":
        k = 10 # deafult 10
    elif args.dataset =="office-home":
        k = 7  # deafult 10
    else:
        k = 5 #deafult 10

    labels = np.asarray(torch.cat((all_label_labeled, all_label), 0).numpy())
    labeled_idx = np.asarray(range(len(labels))[0:len(all_label_labeled)])
    unlabeled_idx = np.asarray(range(len(labels))[len(all_label_labeled):])
    with torch.no_grad():
        output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

        Fea = torch.cat((all_fea_labeled, all_fea), 0)
        N=Fea.shape[0]
        X = F.normalize(Fea, dim=1)
        simlarity_matrix = X.matmul(X.transpose(0, 1))
        D, I = torch.topk(simlarity_matrix, k + 1)
        D = D.cpu().numpy()
        I = I.cpu().numpy()

        # Create the graph
        D = D[:, 1:] ** 4
        I = I[:, 1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx, (k, 1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1. / np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class  and apply label propagation
        Z = np.zeros((N, K))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn

        for i in range(K):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] == i)]
            y = np.zeros((N,))
            if not cur_idx.shape[0]==0:
                y[cur_idx] = 1.0 / cur_idx.shape[0]
            else:
                y[cur_idx] = 1.0

            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:, i] = f

        # Handle numberical errors
        Z[Z < 0] = 0

        # Compute the weight for each instance based on the entropy 
        probs_l1 = F.normalize(torch.tensor(Z), 1).numpy()
        probs_l1[probs_l1 < 0] = 0
        # entropy = scipy.stats.entropy(probs_l1.T)
        # weights = entropy / np.log(K)
        # weights = weights / np.max(weights)
        p_labels = np.argmax(probs_l1, 1)

        # Compute the accuracy of pseudolabels for statistical purposes
        correct_idx = (p_labels[unlabeled_idx] == labels[unlabeled_idx])
        acc = correct_idx.mean()

        p_labels[labeled_idx] = labels[labeled_idx]
        # weights[labeled_idx] = 1.0


    log_str = 'LP Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')

    return p_labels[len(all_label_labeled):].astype('int') 

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets,T=1.0):
        log_probs = self.logsoftmax(inputs/T)

        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        # print((targets * log_probs).mean(0).shape)
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss

def cal_acc(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels=labels.cuda()#2020 07 06
            # inputs = inputs
            outputs= netC(netF(inputs))
            # outputs,margin_logits = netC(netF(inputs),labels)
            labels=labels.cpu()#2020 07 06
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent


def get_model(args):
    netF,netC,netD=None,None,None
    if args.net == 'resnet34':
        netF = resnet34(args=args)
        inc = args.bottleneck
        netC=Predictor(num_class=args.class_num,inc=inc,norm_feature=args.norm_feature,temp=args.temp)
    # elif args.net == 'resnet50':
    #     netF = resnet50(args=args)
    #     inc = args.bottleneck
    #     netC=Predictor_deep(num_class=args.class_num,inc=inc,norm_feature=args.norm_feature,temp=args.temp)
    elif args.net == "alexnet":
        inc = args.bottleneck
        netF = AlexNetBase(bootleneck_dim=inc)
        netC = Predictor(num_class=args.class_num, inc=inc,norm_feature=args.norm_feature,temp=args.temp)
    elif args.net == "vgg":
        inc = args.bottleneck
        netF = VGGBase(bootleneck_dim=inc)
        netC = Predictor(num_class=args.class_num, inc=inc,norm_feature=args.norm_feature,temp=args.temp)
    else:
        raise ValueError('Model cannot be recognized.')

    return  netF,netC,netD


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation')
    parser.add_argument('--gpu_id', type=int, default=1, help="device id to run")
    # parser.add_argument('--net', type=str, default="alexnet", choices=['vgg',"alexnet","resnet34"])
    parser.add_argument('--s', type=str, default="webcam", help="source office_home :Art Clipart Product Real_World")
    parser.add_argument('--t', type=str, default="amazon", help="target  Art Clipart Product Real_World")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch ")
    parser.add_argument('--num', type=int, default=1, help="labeled_data per class. 1: 1-shot, 3: 3-shot")
    parser.add_argument('--train', type=int, default=1, help="if to train")
    # parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    # parser.add_argument('--class_num', type=int, default=31, help="batch_size",choices=[65,31,126])
    parser.add_argument('--worker', type=int, default=8, help="number of workers")
    parser.add_argument('--dataset', type=str, default='Office-31', choices=['office-home', 'multi', 'Office-31'])
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--norm_feature', type=int, default=1, help="random seed")
    parser.add_argument('--par', type=float, default=1)
    parser.add_argument('--temp', type=float, default=0.05)
    parser.add_argument('--im', type=float, default=1)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='SSDA')
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--unlent', type=float, default=1)
    # parser.add_argument('--unl_w', type=float, default=0.5)
    parser.add_argument('--vat_w', type=float, default=1)
    parser.add_argument('--div_w', type=float, default=1)
    parser.add_argument('--max_num_per_class', type=int, default=1, help="max-number per class to select, the MNPC in paper")
    parser.add_argument('--uda', type=int, default=0, help="if to perform unsurpervised DA")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import warnings
    warnings.filterwarnings("ignore")
    current_folder = "./log"


    if args.dataset== "Office-31":
        args.net="alexnet"
        args.class_num=31
        args.batch_size=64
        args.unl_w=0.5

    elif args.dataset== "office-home":
        args.net="vgg"
        args.class_num=65
        args.batch_size=16
        args.unl_w=0.3
    elif args.dataset== "multi":
        args.net="resnet34"
        args.class_num=126
        args.batch_size=64
        args.unl_w=0.3
    else:
        print("We do not have the dataset", args.dataset )


    args.output_dir = osp.join(current_folder, args.output, 'seed' + str(args.seed), args.dataset,args.s+"_lr"+str(args.lr))
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.train==1:
        args.out_file = open(osp.join(args.output_dir, 'log_src_val.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
    else:
        args.out_file = open(osp.join(args.output_dir, 'tar_' + args.s+"2"+args.t+"_lr"+str(args.lr)
                                      + "_unl_ent"+str(args.unlent)+ "_unl_w"+str(args.unl_w)+
                                      "_vat_w"+str(args.vat_w)+ "_div_w"+str(args.div_w)+
                                      "_MNPC"+str(args.max_num_per_class)+"_num"+str(args.num)+'.txt'), 'w')
        test_target(args)
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)
