import argparse
import yaml
import os
import torch
from scape_dataset import ScapeDataset, shape_to_device
from model import DQFMNet
#
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from utils import read_geodist, augment_batch, ICP_rot
from Tools.utils import fMap2pMap, zo_fmap
from diffusion_net.utils import toNP
import torch.nn.functional as F
import potpourri3d as pp3d


def knnsearch_t(x, y):
    # distance = torch.cdist(x.float(), y.float())
    distance = torch.cdist(x.float(), y.float(), compute_mode='donot_use_mm_for_euclid_dist')
    _, idx = distance.topk(k=1, dim=-1, largest=False)
    return idx

def search_t(A1, A2):
    T12 = knnsearch_t(A1, A2)
    T21 = knnsearch_t(A2, A1)
    return T12, T21

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
    test_dataset = ScapeDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                    k_eig=cfg["fmap"]["k_eig"], n_fmap=cfg["fmap"]["n_fmap"], 
                                    with_wks=with_wks, use_cache=True, 
                                    op_cache_dir=op_cache_dir, train=False)

    # test loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=None, shuffle=False)

    # define model
    dqfm_net = DQFMNet(cfg).to(device)
    print(model_path)
    dqfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dqfm_net.eval()

    for i, data in tqdm(enumerate(test_loader)):
        data = shape_to_device(data, device)

        # do iteration
        C12_pred, C21_pred, feat1, feat2, evecs_trans1, evecs_trans2, evecs1, evecs2 = dqfm_net(data)
        Phi1, Phi2 = data["shape1"]['evecs'], data["shape2"]['evecs']
        name1, name2 = data["shape1"]["name"], data["shape2"]["name"]

        save_path_c = save_path + '/C/'
        if not os.path.exists(save_path_c):
            os.makedirs(save_path_c)

        filename_c12 = f'C_{name1}_{name2}.mat'
        c12 = C12_pred.detach().cpu().squeeze(0).numpy()
        c12_dic = {'C': c12}
        sio.savemat(os.path.join(save_path_c, filename_c12), c12_dic)

        save_path_phi = save_path + '/Phi/'
        if not os.path.exists(save_path_phi):
            os.makedirs(save_path_phi)

        filename_phi1 = f'Phi_{name1}.mat'
        Phi1 = toNP(Phi1)
        Phi1_dic = {'Phi': Phi1}
        sio.savemat(os.path.join(save_path_phi, filename_phi1), Phi1_dic)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the eval of DQFM model.")

    parser.add_argument("--config", type=str, default="scape_r", help="Config file name")

    parser.add_argument("--model_path", type=str, default="data/trained_scape/ep_5.pth",
                         help="path to saved model")
    parser.add_argument("--save_path", type=str, default="data/results",
                        help="dir to save C_pred")


    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./config/{args.config}.yaml", "r"))
    eval_net(cfg, args.model_path, args.save_path)
