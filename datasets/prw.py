import os.path as osp
import re

import numpy as np
from scipy.io import loadmat
from collections import OrderedDict

from .base import BaseDataset


class PRW(BaseDataset):
    def __init__(self, root, transforms, split, is_source=False, build_tiny=False):
        self.name = "PRW"
        self.img_prefix = osp.join(root, "frames")
        super(PRW, self).__init__(root, transforms, split, is_source=is_source, build_tiny=build_tiny)

    def _get_cam_id(self, img_name):
        match = re.search(r"c\d", img_name).group().replace("c", "")
        return int(match)

    def _load_queries(self):
        query_info = osp.join(self.root, "query_info.txt")
        with open(query_info, "rb") as f:
            raw = f.readlines()

        queries = []
        for line in raw:
            linelist = str(line, "utf-8").split(" ")
            pid = int(linelist[0])
            x, y, w, h = (
                float(linelist[1]),
                float(linelist[2]),
                float(linelist[3]),
                float(linelist[4]),
            )
            roi = np.array([x, y, x + w, y + h]).astype(np.int32)
            roi = np.clip(roi, 0, None)  # several coordinates are negative
            img_name = linelist[5][:-2] + ".jpg"
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([pid]),
                    "cam_id": self._get_cam_id(img_name),
                }
            )
        return queries

    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        if self.split == "train":
            imgs = loadmat(osp.join(self.root, "frame_train.mat"))["img_index_train"]
        else:
            imgs = loadmat(osp.join(self.root, "frame_test.mat"))["img_index_test"]
        return [img[0][0] + ".jpg" for img in imgs]

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        annotations = []
        imgs = self._load_split_img_names()

        #用500张作subset来debug, 需要重新分配pid
        if self.build_tiny:
            imgs = imgs[:500]
            exist_pids = OrderedDict()

        for img_name in imgs:
            anno_path = osp.join(self.root, "annotations", img_name)
            anno = loadmat(anno_path)
            box_key = "box_new"
            if box_key not in anno.keys():
                box_key = "anno_file"
            if box_key not in anno.keys():
                box_key = "anno_previous"

            rois = anno[box_key][:, 1:]
            ids = anno[box_key][:, 0]
            rois = np.clip(rois, 0, None)  # several coordinates are negative

            assert len(rois) == len(ids)

            rois[:, 2:] += rois[:, :2]
            ids[ids == -2] = 5555  # assign pid = 5555 for unlabeled people
            ids[ids == 932] = 479
            if self.build_tiny:
                for id in ids:
                    exist_pids[id] = id
            #if self.split == "train":
            #    ids[ids != 6100]+=5532
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": rois.astype(np.int32),
                    # FIXME: (training pids) 1, 2,..., 478, 480, 481, 482, 483, 932, 5555
                    "pids": ids.astype(np.int32),
                    "cam_id": self._get_cam_id(img_name),
                }
            )
        if self.build_tiny:
            #debug用tiny时重新分配pid
            for i, (key, value) in enumerate(exist_pids.items()):
                if key != 5555:
                    exist_pids[key] = i
            for anno in annotations:
                for index in range(len(anno["pids"])):
                    anno["pids"][index] = exist_pids[anno["pids"][index]]
        return annotations
