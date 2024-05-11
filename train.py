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

def Random_select_prototype_vector(features, prototype_vector_num):
    print("==> randomly select prototype_vector")
    feature_num = len(features)
    list = random.sample(range(0, feature_num), prototype_vector_num)
    prototype_vector = torch.stack([features[i] for i in list], dim=0)
    print("randomly select", len(prototype_vector), "prototype_vector: finished")
    return prototype_vector

def assign_pseudo_labels(prototype_vector, features):
    print('==> Assign pseudo labels by knn')
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(prototype_vector)
    NearestNeighbors(algorithm='auto', leaf_size=30)
    pseudo_labels = neigh.kneighbors(features, return_distance=False)  # [12936,1]

    m = pseudo_labels
    pseudo_labels = []
    for i in range(len(m)):
        for j in range(len(m[i])):
            pseudo_labels.append(m[i][j])
    num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)

    return pseudo_labels,num_cluster


def Extract_prototype_vector(model, cfg, sour_cluster_loader, device):
    ##############################
    ## Extract prototype vector ##
    ##############################
    # modelpath = 'examples/pretrained/' + datasetname + '/model_best.pth.tar'
    # model.load_state_dict(torch.load(modelpath)['state_dict'])
    with torch.no_grad():
        print('==> Extract prototype vector from', cfg.INPUT.DATASET)
        # cluster_loader = build_cluster_loader(Source_dataset, args.height, args.width,
        #                                  args.batch_size, args.workers, testset=sorted(Source_dataset.train))

        sour_fea_dict = extract_dy_features(cfg, model, sour_cluster_loader, device, is_source=True)
        source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
        source_centers = torch.stack(source_centers,0)
        source_centers = F.normalize(source_centers, dim=1)
    

    return source_centers

def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Creating model and convert dsbn")
    model = SeqNetDa(cfg)
    
    convert_dsbn(model.roi_heads.reid_head)
    model.to(device)

    print("Building dataset")
    transforms = build_transforms(is_train=False)
    dataset_source_train = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train", is_source=True)

    source_classes  = dataset_source_train.num_train_pids
    print("source_classes :"+str(source_classes))

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
    # Create hybrid memory
    memory = HybridMemory(256, source_classes, source_classes,
                            temp=0.05, momentum=0.2).to(device)
    
    # init source domian identity level centroid
    print("==> Initialize source-domain class centroids in the hybrid memory")
    sour_cluster_loader = build_cluster_loader(cfg,dataset_source_train)
    # sour_fea_dict = extract_dy_features(cfg, model, sour_cluster_loader, device, is_source=True)
    # source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    # source_centers = torch.stack(source_centers,0)
    # source_centers = F.normalize(source_centers, dim=1)
    source_centers = Extract_prototype_vector(model, cfg, sour_cluster_loader, device)
    print("source_centers length")
    print()
    print(len(source_centers))
    print(source_centers.shape)
    print("the last one is the feature of 5555, remember don't use it")

    memory.features = source_centers.to(device)

    print(f"memory.features type: {type(memory.features)}")
    print(f"memory.features length: {len(memory.features)}")
    del sour_cluster_loader


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    model.roi_heads.memory = memory
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    start_epoch = 0
    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1

    print("Creating output folder")
    output_dir = cfg.OUTPUT_DIR
    mkdir(output_dir)
    path = osp.join(output_dir, "config.yaml")
    target_start_epoch = cfg.TARGET_REID_START
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config is saved to {path}")
    tfboard = None
    if cfg.TF_BOARD:
        from torch.utils.tensorboard import SummaryWriter

        tf_log_path = osp.join(output_dir, "tf_log")
        mkdir(tf_log_path)
        tfboard = SummaryWriter(log_dir=tf_log_path)
        print(f"TensorBoard files are saved to {tf_log_path}")

    print("Start training")
    del dataset_source_train
    transforms = build_transforms(is_train=True)
    dataset_source_train = build_dataset(cfg.INPUT.DATASET, cfg.INPUT.DATA_ROOT, transforms, "train", is_source=True)
    dataset_target_train = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "train", is_source=False)

    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):

            
        if (epoch>=target_start_epoch):
            

            # init target domain instance level features
            # we can't use target domain GT detection box feature to init, this is only for measuring the upper bound of cluster performance
            #for dynamic clustering method, we use the proposal after several epoches for first init, moreover, we'll update the memory with proposal before each epoch
            print("==> Initialize target-domain instance features in the hybrid memory")
            transforms = build_transforms(is_train=False)
            dataset_target_train = build_dataset(cfg.INPUT.TDATASET, cfg.INPUT.TDATA_ROOT, transforms, "train", is_source=False)
            tgt_cluster_loader = build_cluster_loader(cfg,dataset_target_train)
            if epoch==target_start_epoch:
                target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(cfg, model, tgt_cluster_loader, device, is_source=False)
            else:
                target_features = memory.features[source_classes:].data.cpu().clone()
                #target_features = memory.features[source_classes:source_classes+len(sorted_keys)].data.cpu().clone()
                target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(cfg, model, tgt_cluster_loader, device, is_source=False, memory_proposal_boxes=img_proposal_boxes, memory_target_features=target_features)
            
            sorted_keys = sorted(target_features.keys())

            print("target_features instances :"+str(len(sorted_keys)))
            target_features = torch.cat([target_features[name] for name in sorted_keys], 0)
            target_features = F.normalize(target_features, dim=1).to(device)

            negative_fea = torch.cat([negative_fea[name] for name in sorted(negative_fea.keys())], 0)
            negative_fea = F.normalize(negative_fea, dim=1).to(device)
            print("hard negative instances :"+str(len(negative_fea)))
            
            prototype_vector_num = 500

            prototype_vector = Random_select_prototype_vector(target_features, prototype_vector_num)
            prototype_vector = prototype_vector.cpu()
            target_features = target_features.cpu()
            pseudo_labels, num_cluster = assign_pseudo_labels(prototype_vector,target_features)
            pseudo_labels_source, num_cluster_source = assign_pseudo_labels(source_centers, target_features)
            pseudo_labels = SelectCluster(pseudo_labels,pseudo_labels_source,num_cluster,num_cluster_source,dataset_target_train)

            @torch.no_grad()
            def generate_cluster_features(labels, features, vector):
                centers = collections.defaultdict(list)
                for j in range(len(vector)):
                    centers[j].append(vector[j])
                    
                for i, label in enumerate(labels):
                    if label == -1:
                        continue
                    centers[labels[i]].append(features[i])

                centers = [
                    torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                ]

                centers = torch.stack(centers, dim=0)
                return centers

            cluster_features = generate_cluster_features(pseudo_labels, target_features, prototype_vector)
            cluster_features_source = generate_cluster_features(pseudo_labels_source, target_features,source_centers)
            source_centers = copy.deepcopy(cluster_features_source)  # update
            index_count = 0
            print(f"cluster_features length: {len(cluster_features)}")
            print(f"cluster_features_source length: {len(cluster_features_source)}")

            del tgt_cluster_loader
            for i, anno in enumerate(dataset_target_train.annotations):
                boxes_nums = len(img_proposal_boxes[anno["img_name"]])
                anno["pids"]=torch.zeros(boxes_nums)
                anno["boxes"]=img_proposal_boxes[anno["img_name"]]
                for j in range(boxes_nums):
                    index = sorted_keys.index(anno["img_name"]+"_"+str(j))
                    # print(index)
                    label = pseudo_labels[index] 
                    
                    anno["pids"][j] = index_count+source_classes+1
                    index_count += 1
                dataset_target_train.annotations[i] = anno


            memory_source  = ClusterMemory(0, num_cluster_source , temp=0.05,
                                   momentum=0.2, ).cuda()
            memory_source.features = F.normalize(cluster_features_source, dim=1).cuda()
            model.roi_heads.memory_source = memory_source
            
            memory_index  = ClusterMemory(0, len(dataset_target_train), temp=0.05,
                                   momentum=0.2, ).cuda()
            memory_index.features = F.normalize(target_features, dim=1).cuda()
            model.roi_heads.memory_index = memory_index

            memory.features = torch.cat((source_centers, target_features), dim=0).cuda()
            memory.features = torch.cat((memory.features, negative_fea), dim=0).cuda()

            # hard_negative cases are assigned with unused labels
            pseudo_labels = torch.tensor(pseudo_labels)
            memory.labels = (torch.cat((torch.arange(source_classes), pseudo_labels , torch.arange(len(negative_fea))+pseudo_labels.max()+1))).to(device)
            memory.num_samples = memory.features.shape[0]

        else:
            memory.labels = (torch.arange(source_classes)).to(device)
            memory.num_samples = source_classes


        assert len(memory.labels) == len(memory.features)

        train_loader_s, train_loader_t = build_train_loader_da_dy_cluster(cfg, dataset_source_train, dataset_target_train)

        train_one_epoch_da(cfg, model, optimizer, train_loader_s, train_loader_t, device, epoch, tfboard)
        lr_scheduler.step()

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            evaluate_performance(
                model,
                gallery_loader,
                query_loader,
                device,
                use_gt=cfg.EVAL_USE_GT,
                use_cache=cfg.EVAL_USE_CACHE,
                use_cbgm=cfg.EVAL_USE_CBGM,
            )

        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 1:
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    'amp': amp.state_dict()
                },
                osp.join(output_dir, f"epoch_{epoch}.pth"),
            )

    if tfboard:
        tfboard.close()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time {total_time_str}")

# generate clean and noise samples
def generate_pseudo_labels(cluster_id, num, dataset_target):
    labels = []
    outliers = 0
    assert len(dataset_target) == len(cluster_id)
    outlier_label = torch.zeros(len(dataset_target), len(dataset_target))
    for i, id in enumerate(cluster_id):
        if id != 5555:
            labels.append(id)
        else:
            outlier_label[i, i] = 1
            labels.append(num + outliers)
            outliers += 1
    return torch.Tensor(labels).long(), outlier_label

def SelectCluster(pseudo_labels,pseudo_labels_prev,num_cluster_label,num_cluster_label_prev,dataset_target):
    pseudo_labels_noNeg1, outlier_oneHot = generate_pseudo_labels(pseudo_labels,num_cluster_label,dataset_target)#number label and one-hot label
    pseudo_labels_noNeg1_prev, outlier_oneHot_prev = generate_pseudo_labels(pseudo_labels_prev, num_cluster_label_prev,dataset_target)

    # ---------------------------------------
    N = pseudo_labels_noNeg1.size(0)

    label_sim = pseudo_labels_noNeg1.expand(N, N).eq(pseudo_labels_noNeg1.expand(N,N).t()).float()  # if label_sim[0]=[1,0,0,1,0],it means sample0 and smaple3 are assigned the same pseudo label
    label_sim_prev = pseudo_labels_noNeg1_prev.expand(N, N).eq(pseudo_labels_noNeg1_prev.expand(N, N).t()).float()  # so label_sim_1[0] may be = [1,0,0,0,1]

    label_sim_new = label_sim - outlier_oneHot
    label_sim_new_prev = label_sim_prev - outlier_oneHot_prev

    label_share = torch.min(label_sim_new,label_sim_new_prev)  # label_sim_new[0]means the first sample's cluster result and its neighbors and so is label_sim_mew_1[0],so we use torch.min
    uncer = label_share.sum(-1) / label_sim.sum(-1)  # model union model_prev/ model

    a = torch.le(uncer, 0.3).type(torch.uint8)  # if uncer<0.8,return true
    b = torch.gt(label_sim.sum(-1), 1).type(torch.uint8)  # to check whether when we share the cluster result,any sample degrade to an outlier
    index_zero = a * b  # value 0 means clean
    index_zero = torch.nonzero(index_zero)  # now we get not clean samples' index
    np.array(pseudo_labels)[index_zero.numpy()] = -1

    return pseudo_labels

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
