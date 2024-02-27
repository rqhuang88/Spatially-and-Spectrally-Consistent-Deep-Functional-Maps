import argparse
import yaml
import os
import torch
from scape_dataset import ScapeDataset, shape_to_device
# from dt4d_dataset import ScapeDataset, shape_to_device
# from garmcap_dataset import ScapeDataset, shape_to_device
from model import DQFMNet
from utils import DQFMLoss, augment_batch, augment_batch_sym
from sklearn.neighbors import NearestNeighbors
#
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from diffusion_net.utils import toNP
import  torch.nn.functional as F

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    bs, m, n = x.size(0), x.size(1), y.size(1)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(bs, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(bs, n, m).transpose(1, 2)
    dist = xx + yy - 2 * torch.bmm(x, y.transpose(1, 2))
    dist = dist.clamp(min=1e-12).sqrt()
    return dist


# It is equal to Tij = knnsearch(j, i) in Matlab
def knnsearch(x, y, alpha):
    distance = torch.cdist(x.float(), y.float())
    # distance = euclidean_dist(x, y)
    output = F.softmax(-alpha*distance, dim=-1)
    # _, idx = distance.topk(k=k, dim=-1)
    return output


def convert_C(Phi1, Phi2, A1, A2, alpha):
    Phi1, Phi2 = Phi1[:, :50].unsqueeze(0), Phi2[:, :50].unsqueeze(0)
    D1 = torch.bmm(Phi1, A1)
    D2 = torch.bmm(Phi2, A2)
    T12 = knnsearch(D1, D2, alpha)
    T21 = knnsearch(D2, D1, alpha)
    C12_new = torch.bmm(torch.pinverse(Phi2), torch.bmm(T21, Phi1))
    C21_new = torch.bmm(torch.pinverse(Phi1), torch.bmm(T12, Phi2))

    return C12_new, C21_new


def train_net(cfg):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path_train = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_train"])
    dataset_path_test = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])

    save_dir_name = f'trained_{cfg["dataset"]["name"]}'
    model_save_path = os.path.join(base_path, f"data/{save_dir_name}/ep" + "_{}.pth")
    if not os.path.exists(os.path.join(base_path, f"data/{save_dir_name}/")):
        os.makedirs(os.path.join(base_path, f"data/{save_dir_name}/"))

    # decide on the use of WKS descriptors
    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    # create dataset
    # standard structured (source <> target) vts dataset
    if cfg["dataset"]["type"] == "vts":
        train_dataset = ScapeDataset(dataset_path_train, name=cfg["dataset"]["name"]+"-"+cfg["dataset"]["subset"],
                                     k_eig=cfg["fmap"]["k_eig"],
                                     n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                     with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                     use_cache=True, op_cache_dir=op_cache_dir, train=True)

        test_dataset = ScapeDataset(dataset_path_test, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                    k_eig=cfg["fmap"]["k_eig"],
                                    n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                    with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                    use_cache=True, op_cache_dir=op_cache_dir, train=False)
    # else not implemented
    else:
        raise NotImplementedError("dataset not implemented!")

    # data loader
    # sampler_train = torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(len(train_dataset) * 0.6))
#     sampler_test = torch.utils.data.RandomSampler(test_dataset, replacement=True, num_samples=int(len(test_dataset) * 0.5))
    
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, sampler=sampler_train)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, sampler=sampler_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=False)

    # define model
    dqfm_net = DQFMNet(cfg).to(device)
    lr = float(cfg["optimizer"]["lr"])
    optimizer = torch.optim.Adam(dqfm_net.parameters(), lr=lr, betas=(cfg["optimizer"]["b1"], cfg["optimizer"]["b2"]))
    criterion = DQFMLoss(w_gt=cfg["loss"]["w_gt"],
                         w_ortho=cfg["loss"]["w_ortho"],
                         w_bij=cfg["loss"]["w_bij"],
                         w_res=cfg["loss"]["w_res"]).to(device)

    # Training loop
    print("start training")
    total_iter = len(train_loader)

    alpha_list = np.linspace(cfg["loss"]["min_alpha"], cfg["loss"]["max_alpha"]+1, cfg["training"]["epochs"])
    eval_best_loss = 1e10
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        if epoch % cfg["optimizer"]["decay_iter"] == 0:
            lr *= cfg["optimizer"]["decay_factor"]
            print(f"Decaying learning rate, new one: {lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        loss_sum, loss_ortho_sum, loss_bij_sum, loss_res_sum = 0, 0, 0, 0
        iterations = 0
        alpha_i = alpha_list[epoch-1]
        dqfm_net.train()
        for i, data in tqdm(enumerate(train_loader)):
            data = shape_to_device(data, device)
            # data augmentation (if we have wks descriptors we use sym augmentation)
            if True and with_wks is None:
                # data = augment_batch(data, rot_x=180, rot_y=180, rot_z=180,
                #                      std=0.01, noise_clip=0.05,
                #                      scale_min=0.9, scale_max=1.1)
                data = augment_batch(data, rot_x=0, rot_y=180, rot_z=0,
                                     std=0.01, noise_clip=0.05,
                                     scale_min=0.9, scale_max=1.1)

            # prepare iteration data
            C12_gt, C21_gt = data["C12_gt"].unsqueeze(0), data["C21_gt"].unsqueeze(0)
            C12_pred, C21_pred, Q_pred, feat1, feat2, evecs_trans1, evecs_trans2, evecs1, evecs2 = dqfm_net(data)
            
            A1 = torch.bmm(evecs_trans1.unsqueeze(0), feat1)
            A2 = torch.bmm(evecs_trans2.unsqueeze(0), feat2)
            C12_pred_new, C21_pred_new = convert_C(evecs1, evecs2, A1, A2, alpha_i)
            # C12_pred_new, C21_pred_new = C12_pred, C21_pred

            loss, loss_gt_old, loss_gt, loss_ortho, loss_bij, loss_res = criterion(C12_gt, C21_gt, C12_pred.to(device), C21_pred.to(device), C12_pred_new.to(device), C21_pred_new.to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log
            # iterations += 1
            # loss_sum += loss
            # loss_ortho_sum += loss_ortho
            # loss_bij_sum += loss_bij
            # loss_res_sum += loss_res

#             log_batch = (i + 1) % cfg["misc"]["log_interval"] == 0
#             if log_batch:
#                 print(f"epoch:{epoch}, loss:{loss_sum/iterations}, gt_old_loss:{loss_gt_old_sum/iterations}, "
#                       f"ortho_loss:{loss_ortho_sum/iterations}, bij_loss:{loss_bij_sum/iterations}, "
#                       f"res_loss:{loss_res_sum/iterations}")
        
#         print(f"epoch:{epoch}, loss:{loss_sum/iterations}, "
#                       f"ortho_loss:{loss_ortho_sum/iterations}, bij_loss:{loss_bij_sum/iterations}, "
#                       f"res_loss:{loss_res_sum/iterations}")

        with torch.no_grad():
            eval_loss = 0
            eval_ortho_loss = 0
            eval_res_loss = 0
            val_iters = 0
            dqfm_net.eval()
            for i, data in tqdm(enumerate(test_loader)):
                data = shape_to_device(data, device)
                optimizer.zero_grad()
                C12_gt, C21_gt = data["C12_gt"].unsqueeze(0), data["C21_gt"].unsqueeze(0)
                C12_pred, C21_pred, Q_pred, feat1, feat2, evecs_trans1, evecs_trans2, evecs1, evecs2 = dqfm_net(data)
                
                A1 = torch.bmm(evecs_trans1.unsqueeze(0), feat1)
                A2 = torch.bmm(evecs_trans2.unsqueeze(0), feat2)
                C12_pred_new, C21_pred_new = convert_C(evecs1, evecs2, A1, A2, alpha_i)
                # C12_pred_new, C21_pred_new = C12_pred, C21_pred

                loss, loss_gt_old, loss_gt, loss_ortho, loss_bij, loss_res = criterion(C12_gt, C21_gt, C12_pred.to(device), C21_pred.to(device), C12_pred_new.to(device), C21_pred_new.to(device))
                val_iters += 1
                eval_loss += loss
                eval_ortho_loss += loss_ortho
                eval_res_loss += loss_res
                
            print(f"epoch:{epoch}, val_loss:{eval_loss/val_iters}, val_ortho_loss:{eval_ortho_loss/val_iters}, val_res_loss:{eval_res_loss/val_iters}")
            

        # save model
        if (epoch + 1) % cfg["misc"]["checkpoint_interval"] == 0:
            torch.save(dqfm_net.state_dict(), model_save_path.format(epoch))
            
        if eval_loss <= eval_best_loss:
            eval_best_loss = eval_loss
            torch.save(dqfm_net.state_dict(), model_save_path.format('val_best'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the training of DQFM model.")
    parser.add_argument("--config", type=str, default="smal_r", help="Config file name")

    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    train_net(cfg)
