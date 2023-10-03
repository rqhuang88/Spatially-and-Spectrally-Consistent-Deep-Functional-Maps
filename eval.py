import argparse
import yaml
import os
import torch
from scape_dataset_eval import ScapeDataset, shape_to_device
# from dt4d_dataset import ScapeDataset, shape_to_device
from model_eval import DQFMNet
#
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from utils import read_geodist, augment_batch, augment_batch_sym
from Tools.utils import fMap2pMap, zo_fmap
from diffusion_net.utils import toNP
import torch.nn.functional as F

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
    # dist = dist.clamp(min=1e-12).sqrt() 
    return dist


# It is equal to Tij = knnsearch(j, i) in Matlab
def knnsearch(x, y, alpha):
    distance = euclidean_dist(x, y)
    output = F.softmax(-alpha*distance, dim=-1)
    # _, idx = distance.topk(k=k, dim=-1)
    return output


def convert_C(C12, Phi1, Phi2, alpha):
    Phi1, Phi2 = Phi1[:, :80].unsqueeze(0), Phi2[:, :80].unsqueeze(0)
    T21 = knnsearch(torch.bmm(Phi2, C12), Phi1, alpha)
    C12_new = torch.bmm(torch.pinverse(Phi2), torch.bmm(T21, Phi1))

    return C12_new


def eval_geodist(cfg, shape1, shape2, T):
    path_geodist_shape2 = os.path.join(cfg['dataset']['root_geodist'], shape2['name']+'.mat')
    MAT_s = sio.loadmat(path_geodist_shape2)

    G_s, SQ_s = read_geodist(MAT_s)

    n_s = G_s.shape[0]
    # print(SQ_s[0])
    if 'vts' in shape1:
        phi_t = shape1['vts']
        phi_s = shape2['vts']
    elif 'gt' in shape1:
        phi_t = np.arange(shape1['xyz'].shape[0])
        phi_s = shape1['gt']
    else:
        raise NotImplementedError("cannot find ground-truth correspondence for eval")

    # find pairs of points for geodesic error
    pmap = T
    ind21 = np.stack([phi_s, pmap[phi_t]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[n_s, n_s])

    errC = np.take(G_s, ind21) / SQ_s
    print('{}-->{}: {:.4f}'.format(shape1['name'], shape2['name'], np.mean(errC)))
    return errC

def eval_net(args, model_path, save_path):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])

    # decide on the use of WKS descriptors
    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    # create dataset
    train_dataset = ScapeDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                    k_eig=cfg["fmap"]["k_eig"],
                                    n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                    with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                    use_cache=True, op_cache_dir=op_cache_dir, train=True)
#     if cfg["dataset"]["type"] == "vts":
#         test_dataset = ScapeDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
#                                     k_eig=cfg["fmap"]["k_eig"],
#                                     n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
#                                     with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
#                                     use_cache=True, op_cache_dir=op_cache_dir, train=False)

#     elif cfg["dataset"]["type"] == "gt":
#         test_dataset = ShrecDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
#                                     k_eig=cfg["fmap"]["k_eig"],
#                                     n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
#                                     with_wks=with_wks,
#                                     use_cache=True, op_cache_dir=op_cache_dir, train=False)

    # else:
    #     raise NotImplementedError("dataset not implemented!")

    # test loader
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=False)

    # define model
    dqfm_net = DQFMNet(cfg).to(device)
    print(model_path)
    dqfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dqfm_net.eval()

    to_save_list = []
    errs = []

    for i, data in tqdm(enumerate(test_loader)):
        data = shape_to_device(data, device)

        # data augmentation (if using wks descriptors augment with sym)
        # if with_wks is None:
        #     data = augment_batch(data, rot_x=180, rot_y=180, rot_z=180,
        #                          std=0.01, noise_clip=0.05,
        #                          scale_min=0.9, scale_max=1.1)
        # elif "with_sym" in cfg["dataset"] and cfg["dataset"]["with_sym"]:
        #     data = augment_batch_sym(data, rand=False)

        # prepare iteration data

        # do iteration
        C_pred, Q_pred = dqfm_net(data)
        Phi1, Phi2 = data["shape1"]['evecs'], data["shape2"]['evecs']

        # check rank
        # print(feat1.shape)
        # feat1, feat2 = feat1.cpu().numpy(), feat2.cpu().numpy()
        # rank1, rank2 = torch.linalg.matrix_rank(feat1.squeeze()), torch.linalg.matrix_rank(feat2.squeeze())
        #print([rank1, rank2])

        # save maps
        name1, name2 = data["shape1"]["name"], data["shape2"]["name"]


        save_path_c = save_path + '/C/'
        if not os.path.exists(save_path_c):
            os.makedirs(save_path_c)

        filename_c12 = f'C_{name1}_{name2}.mat'
        c12 = C_pred.detach().cpu().squeeze(0).numpy()
        c12_dic = {'C': c12}
        sio.savemat(os.path.join(save_path_c, filename_c12), c12_dic)

        # compute geodesic error (transpose C12 to get C21, and thus T12)
        shape1, shape2 = data["shape1"], data["shape2"]

        save_path_phi = save_path + '/Phi/'
        if not os.path.exists(save_path_phi):
            os.makedirs(save_path_phi)

        filename_phi1 = f'Phi_{name1}.mat'
        Phi1 = toNP(shape1['evecs'])
        Phi1_dic = {'Phi': Phi1}
        sio.savemat(os.path.join(save_path_phi, filename_phi1), Phi1_dic)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the eval of DQFM model.")

    parser.add_argument("--config", type=str, default="smal_r", help="Config file name")

    parser.add_argument("--model_path", type=str, default="data/trained_scape/ep_5.pth",
                         help="path to saved model")
    parser.add_argument("--save_path", type=str, default="data/results",
                        help="dir to save C_pred")


    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    eval_net(cfg, args.model_path, args.save_path)
