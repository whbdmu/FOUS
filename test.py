#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:12:47 2024

@author: huibing
"""

import argparse
import datetime
import os.path as osp
import time
from sklearn.neighbors import NearestNeighbors
import random
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import copy
from datasets import build_test_loader, build_train_loader_da, build_dataset,build_train_loader_da_dy_cluster,build_cluster_loader
from utils.transforms import build_transforms
from defaults import get_default_cfg
from engine import evaluate_performance, train_one_epoch_da, crop_image
from models.seqnet_da import SeqNetDa
from utils.utils import mkdir, resume_from_ckpt, save_on_master, set_random_seed
from apex import amp
import torch.nn.functional as F
from ReLL.evaluators import extract_features, extract_dy_features
from ReLL.models.hm import HybridMemory
import collections
from ReLL.models.cm import ClusterMemory


def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model")
    model = SeqNetDa(cfg)
    
    convert_dsbn(model.roi_heads.reid_head)
    model.to(device)

    print("Building dataset")
    transforms = build_transforms(is_train=False)
    dataset_source_train = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train", is_source=True)

    print("Loading test data")
    gallery_loader, query_loader = build_test_loader(cfg)

    if args.eval:
        assert args.ckpt, "--ckpt must be specified when --eval enabled"
        resume_from_ckpt(args.ckpt, model)
        dataset_target_train = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "train", is_source=False)
        tgt_cluster_loader = build_cluster_loader(cfg, dataset_target_train)
        model.eval()
        evaluate_performance(
            model,
            gallery_loader,
            query_loader,
            device,
            use_gt=cfg.EVAL_USE_GT,
            use_cache=cfg.EVAL_USE_CACHE,
            use_cbgm=cfg.EVAL_USE_CBGM,
        )
        exit(0)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", default='configs/prw_da.yaml', help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    parser.add_argument('--local_rank', default=-1, type=int)

    args = parser.parse_args()
    main(args)
