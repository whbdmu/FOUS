#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:12:47 2024

@author: huibing
"""
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
from spcl.models.dsbn import convert_dsbn
from spcl.models.hm import HybridMemory
from spcl.utils.faiss_rerank import compute_jaccard_distance,update_target_memory
import torch.nn.functional as F
from spcl.evaluators import Evaluator, extract_features,extract_dy_features
from sklearn.cluster import DBSCAN
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
    torch.cuda.set_device(device)
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
    memory_hm = HybridMemory(256, source_classes, source_classes,
                            temp=0.05, momentum=0.2).to(device)

    # init source domian identity level centroid
    print("==> Initialize source-domain class centroids in the hybrid memory")
    sour_cluster_loader = build_cluster_loader(cfg,dataset_source_train)
    # sour_fea_dict = extract_dy_features(cfg, model, sour_cluster_loader, device, is_source=True)
    # source_centers = [torch.cat(sour_fea_dict[pid],0).mean(0) for pid in sorted(sour_fea_dict.keys())]
    # source_centers = torch.stack(source_centers,0)
    # source_centers = F.normalize(source_centers, dim=1)
    source_centers = Extract_prototype_vector(model, cfg, sour_cluster_loader, device)
    prototype_vector_source = copy.deepcopy(source_centers)
    memory_hm.features = source_centers.cuda()
    model.roi_heads.memory_hm = memory_hm
    print("source_centers length")
    print(len(prototype_vector_source))
    print(prototype_vector_source.shape)

    del sour_cluster_loader

    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg.SOLVER.BASE_LR,
        momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=0.1
    )

    if args.resume:
        assert args.ckpt, "--ckpt must be specified when --resume enabled"
        start_epoch = resume_from_ckpt(args.ckpt, model, optimizer, lr_scheduler) + 1
    start_epoch = 0
    

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
        
        print(epoch)            
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
                target_features = memory_instance.features.data.cpu().clone()
                #target_features = memory.features[source_classes:source_classes+len(sorted_keys)].data.cpu().clone()
                target_features, img_proposal_boxes, negative_fea, positive_fea = extract_dy_features(cfg, model, tgt_cluster_loader, device, is_source=False, memory_proposal_boxes=img_proposal_boxes, memory_target_features=target_features)
            
            sorted_keys = sorted(target_features.keys())
            print("target_features instances :"+str(len(sorted_keys)))
            target_features = torch.cat([target_features[name] for name in sorted_keys], 0)
            target_features = F.normalize(target_features, dim=1)

            prototype_vector_num = 500
            prototype_vector = Random_select_prototype_vector(target_features, prototype_vector_num)
            prototype_vector = prototype_vector.cpu()
            target_features = target_features.cpu()
            pseudo_labels, num_cluster = assign_pseudo_labels(prototype_vector, target_features)
            pseudo_labels_source, num_cluster_source = assign_pseudo_labels(prototype_vector_source, target_features)
            pseudo_labels = SelectCluster(pseudo_labels,pseudo_labels_source,num_cluster,num_cluster_source,dataset_target_train)
            print(num_cluster,num_cluster_source)
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
            cluster_features_source = generate_cluster_features(pseudo_labels_source, target_features, prototype_vector_source)
            prototype_vector_source = copy.deepcopy(cluster_features_source)  # update
            
            print(f"cluster_features length: {len(cluster_features)}")
            print(f"cluster_features_source length: {len(cluster_features_source)}")
            print(f"pseudo_labels length: {len(pseudo_labels)}")
            print(f"pseudo_labels_source length: {len(pseudo_labels_source)}")
            del tgt_cluster_loader
            index_count = 0
            for i, anno in enumerate(dataset_target_train.annotations):
                boxes_nums = len(img_proposal_boxes[anno["img_name"]])
                anno["pids"]=torch.zeros(boxes_nums,3)
                anno["boxes"]=img_proposal_boxes[anno["img_name"]]
                for j in range(boxes_nums):
                    index = sorted_keys.index(anno["img_name"]+"_"+str(j))
                    # print(index)
                    label_source = pseudo_labels_source[index] 
                    label = pseudo_labels[index] 

                    anno["pids"][j][0] = label_source
                    anno["pids"][j][1] = label
                    anno["pids"][j][2] = index_count+1

                    index_count += 1
                dataset_target_train.annotations[i] = anno

            """             
            for i, anno in enumerate(dataset_target_train.annotations):
                boxes_nums = len(img_proposal_boxes[anno["img_name"]])
                anno["pids"]=torch.zeros(boxes_nums)
                anno["boxes"]=img_proposal_boxes[anno["img_name"]]
                for j in range(boxes_nums):
                    index = sorted_keys.index(anno["img_name"]+"_"+str(j))
                    # print(index)
                    
                    anno["pids"][j] = label
                dataset_target_train.annotations[i] = anno 
            """

            # Create hybrid memory
            memory  = ClusterMemory(256, num_cluster, source_classes, temp=0.05,
                        momentum=0.2,).to(device)
            memory.features = F.normalize(cluster_features, dim=1).to(device)
            model.roi_heads.memory = memory

            memory_source  = ClusterMemory(256, num_cluster_source, source_classes, temp=0.05,
                        momentum=0.2,).to(device)
            memory_source.features = F.normalize(cluster_features_source, dim=1).to(device)
            model.roi_heads.memory_source = memory_source

            memory_instance  = ClusterMemory(256, len(dataset_target_train), source_classes, temp=0.05,
                                   momentum=0.2, ).cuda()
            memory_instance.features = target_features.to(device)
            model.roi_heads.memory_instance = memory_instance 

            pseudo_labels = torch.tensor(pseudo_labels)+source_classes+1
            pseudo_labels_source = torch.tensor(pseudo_labels_source)+source_classes+1
 
            memory_hm.labels = (torch.arange(source_classes)).to(device)
            memory_hm.num_samples = source_classes

        else:
            memory_hm.labels = (torch.arange(source_classes)).to(device)
            memory_hm.num_samples = source_classes


        train_loader_s, train_loader_t = build_train_loader_da_dy_cluster(cfg, dataset_source_train, dataset_target_train)

        train_one_epoch_da(cfg, model, optimizer, train_loader_s, train_loader_t, device, epoch, tfboard)
        lr_scheduler.step()

        if (epoch + 1) % cfg.EVAL_PERIOD == 0 or epoch == cfg.SOLVER.MAX_EPOCHS - 5:
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
    outlier_label = torch.zeros(len(cluster_id), len(cluster_id))
    for i, id in enumerate(cluster_id):
        if id != -1 and id != 5555:
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
    parser.add_argument("--cfg", dest="cfg_file", default='/home/linfeng/REID/FOUS/base_FOUS/configs/prw_da.yaml', help="Path to configuration file.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the performance of a given checkpoint."
    )
    parser.add_argument(
        "--resume", default=True, action="store_true", help="Resume from the specified checkpoint."
    )
    parser.add_argument("--ckpt", help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    parser.add_argument('--local_rank', default=-1, type=int)

    args = parser.parse_args()
        # 将输入导入日志
    def make_print_to_log(path='./',log_interval = 5.0):
        '''
        path, it is a path for save your log about fuction print
        example:
        use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
        :return:
        '''
        import sys
        import os
        import sys
        import datetime
        import atexit

        class Logger(object):
            def __init__(self, filename="Default.log", path="./run", log_interval=5.0):
                self.terminal = sys.stdout
                self.path = os.path.join(path, filename)
                self.log_interval = log_interval
                self.last_log_time = time.time()
                self.buffer = ""

            def write(self, message):
                self.terminal.write(message)
                self.buffer += message
                current_time = time.time()
                if current_time - self.last_log_time >= self.log_interval:
                    with open(self.path, "a", encoding='utf8') as log_file:
                        log_file.write(self.buffer)
                        log_file.flush()  # 刷新日志文件
                    self.buffer = ""
                    self.last_log_time = current_time

            def flush(self):
                pass
            
            @atexit.register
            def write_last_output():
                if logger.buffer:
                    with open(logger.path, "a", encoding='utf8') as log_file:
                        log_file.write(logger.buffer)
                        log_file.flush()  # 刷新日志文件

        
        t = time.strftime("-%Y%m%d-%H-%M", time.localtime())  # 时间戳
        logger = Logger(filename='log' + t + '.log',path=path,log_interval=log_interval)  
        sys.stdout = logger

    make_print_to_log(path='./run',log_interval = 120)

    main(args)
