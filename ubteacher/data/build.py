# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import operator
import json
import torch.utils.data
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
)
from ubteacher.data.common import (
    AspectRatioGroupedSemiSupDatasetTwoCrop,
)


def make_unlabel_dicts():
    import os
    import cv2

    unlabel_dicts = []
    prefix = '/workspace/unbiased-teacher/datasets/koreandramaep01/new/'
    imgs = os.listdir(prefix)
    for img in imgs:
        im = cv2.imread(prefix+img)
        temp={}
        temp['file_name'] = prefix+img
        temp['height'] = im.shape[0]
        temp['width'] = im.shape[1]
        temp['image_id'] = img
        temp['annotations'] = [{'iscrowd': 0, 'bbox': [0,0,0,0], 'category_id':0, 'bbox_mode':0}]
        unlabel_dicts.append(temp)
    return unlabel_dicts


"""
This file contains the default logic to build a dataloader for training or testing.
"""

def selecet_label(SupPercent):
    from collections import Counter
    with open('/workspace/unbiased-teacher/fixed_inference/coco_instances_results.json', 'r') as f:
        data = f.read()
    data = eval(data)
    dic = {}
    image_id=0
    temp=0.0
    count=0
    for x in data:
        if x['image_id'] in [616, 619]:
            continue
        if x['image_id'] != image_id:
            dic[image_id]=temp/count
            temp=0.0
            count=0
        image_id=x['image_id']
        temp+=x['score']
        count+=1
    dic[image_id]=temp/count
    num_label = int(len(dic)*(SupPercent/100))
    new_dic = Counter(dic)
    use_labels = sorted(new_dic, key=new_dic.get, reverse=True)[:num_label]
    return use_labels


def select_label_by_low_map(SupPercent):
    from collections import Counter
    import os
    with open('./coco/mAP.txt', 'r') as f:
        data = f.read()
    data = data.split('\n')
    image_ids = os.listdir('./coco/detection-results/')
    image_ids = sorted(image_ids)
    mAP = {int(k.split('.')[0]):float(v) for k, v in zip(image_ids, data)}
    new_mAP = Counter(mAP)
    num_label = int(len(data)*(SupPercent/100))
    use_labels_pred = sorted(new_mAP, key=new_mAP.get, reverse=False)[:num_label]
    category_keys = list(new_mAP.keys())
    use_labels = [category_keys.index(x) for x in use_labels_pred]
    use_labels = sorted(use_labels)
    print('use labels :', use_labels)
    return use_labels


def select_label_by_high_map_not_map100(SupPercent):
    from collections import Counter
    import os
    with open('/workspace/unbiased-teacher/coco/mAP.txt', 'r') as f:
        data = f.read()
    data = data.split('\n')
    image_ids = os.listdir('/workspace/unbiased-teacher/coco/detection-results/')
    image_ids = sorted(image_ids)
    mAP = {int(k.split('.')[0]):float(v) for k, v in zip(image_ids, data)}
    new_mAP = Counter(mAP)
    num_label = int(len(data)*(SupPercent/100))
    num_100 = len([x for x in new_mAP.values() if x==100])
    use_labels_pred = sorted(new_mAP, key=new_mAP.get, reverse=True)[num_100:num_100+num_label]
    category_keys = list(new_mAP.keys())
    use_labels = [category_keys.index(x) for x in use_labels_pred]
    use_labels = sorted(use_labels)
    print('use labels :', use_labels)
    return use_labels



def select_label_by_person(SupPercent):
    from collections import Counter
    import os
    import pickle
    with open('./coco/pred.pkl', 'rb') as f:
        pred = pickle.load(f)
    dic_person = {}
    num_person = []
    for x in pred:
        temp = 0
        for i in range(len(x['category_id'])):
            if x['score'][i] > 0.5 and x['category_id'][i] == 1:
                temp += 1
        num_person.append(temp)
        dic_person[x['image_id']] = temp
    
    image_ids = os.listdir('./coco/detection-results/')
    img_ids = [int(k.split('.')[0]) for k in image_ids]
    append_list = list(set(img_ids) - set(dic_person.keys()))
    for k in append_list:
        dic_person[k]=0
    sorted_by_key = sorted(dic_person.items())
    dic_person = {k:v for k,v in sorted_by_key}
    count_person = Counter(dic_person)
    num_label = int(len(image_ids)*(SupPercent/100))
    use_labels = sorted(count_person, key=count_person.get, reverse=False)[:num_label]
    print(use_labels)
    return use_labels


def select_label_by_not_person(SupPercent):
    from collections import Counter
    import os
    import pickle
    with open('./coco/pred.pkl', 'rb') as f:
        pred = pickle.load(f)
    category = {}
    for x in pred:
        temp = 0
        for i in range(len(x['category_id'])):
            if x['score'][i] > 0.5 and x['category_id'][i] != 1:
                temp += 1
        category[x['image_id']] = temp
    
    image_ids = os.listdir('./coco/detection-results/')
    img_ids = [int(k.split('.')[0]) for k in image_ids]
    append_list = list(set(img_ids) - set(category.keys()))
    for k in append_list:
        category[k]=0
    sorted_by_key = sorted(category.items())
    category = {k:v for k,v in sorted_by_key}
    category = Counter(category)
    num_label = int(len(image_ids)*(SupPercent/100))
    use_labels = sorted(category, key=category.get, reverse=True)[:num_label]
    category_keys = list(category.keys())
    use_labels = [category_keys.index(x) for x in use_labels]
    print(use_labels)
    return use_labels


def select_label_by_low_person(SupPercent):
    from collections import Counter
    import os
    import pickle
    with open('./coco/pred.pkl', 'rb') as f:
        pred = pickle.load(f)
    category = {}
    for x in pred:
        temp = 0
        for i in range(len(x['category_id'])):
            if x['score'][i] > 0.5 and x['category_id'][i] == 1:
                temp += 1
        category[x['image_id']] = temp
    
    image_ids = os.listdir('./coco/detection-results/')
    img_ids = [int(k.split('.')[0]) for k in image_ids]
    append_list = list(set(img_ids) - set(category.keys()))
    for k in append_list:
        category[k]=0
    sorted_by_key = sorted(category.items())
    category = {k:v for k,v in sorted_by_key}
    category = Counter(category)
    num_label = int(len(image_ids)*(SupPercent/100))
    use_labels_pred = sorted(category, key=category.get, reverse=False)[:num_label]
    category_keys = list(category.keys())
    use_labels = [category_keys.index(x) for x in use_labels_pred]
    print(use_labels)
    return use_labels

    
def divide_label_unlabel(
    dataset_dicts, SupPercent, random_data_seed, random_data_seed_path
):
    num_all = len(dataset_dicts)
    num_label = int(SupPercent / 100.0 * num_all)
    #num_label = num_all
    # read from pre-generated data seed
    # removed this 2 lines to use custom dataset yeonsik
    # with open(random_data_seed_path) as COCO_sup_file:
    #    coco_random_idx = json.load(COCO_sup_file)
    
    # removed this 1 line and add new 2 lines to use custom dataset yeonsik
    # labeled_idx = np.array(coco_random_idx[str(SupPercent)][str(random_data_seed)])
    # sup_p = 1
    # num_label = int(sup_p / 100. * num_all)
    #labeled_idx = np.random.choice(range(num_all), size=num_label, replace=False)
    labeled_idx = select_label_by_high_map_not_map100(SupPercent)
    #assert labeled_idx.shape[0] == num_label, "Number of READ_DATA is mismatched."
    label_dicts = []
    unlabel_dicts = []
    labeled_idx = set(labeled_idx)
    for i in range(len(dataset_dicts)):
        if i in labeled_idx:
           label_dicts.append(dataset_dicts[i])
        # label_dicts.append(dataset_dicts[i])
        else:
           unlabel_dicts.append(dataset_dicts[i])
    
    # unlabel_dicts = make_unlabel_dicts()
    # print('=========labeled')
    # print(label_dicts[:3])
    # print('=======unlabeled')
    # print(unlabel_dicts[:3])

    return label_dicts, unlabel_dicts


# uesed by supervised-only baseline trainer
def build_detection_semisup_train_loader(cfg, mapper=None):

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Divide into labeled and unlabeled sets according to supervision percentage
    label_dicts, unlabel_dicts = divide_label_unlabel(
        dataset_dicts,
        cfg.DATALOADER.SUP_PERCENT,
        cfg.DATALOADER.RANDOM_DATA_SEED,
        cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
    )
    
    dataset = DatasetFromList(label_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                label_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info("Number of training samples " + str(len(dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# uesed by evaluation
def build_detection_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    dataset = DatasetFromList(dataset_dicts)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


# uesed by unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops(cfg, mapper=None):
    if cfg.DATASETS.CROSS_DATASET:  # cross-dataset (e.g., coco-additional)
        label_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_LABEL,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        unlabel_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_UNLABEL,
            filter_empty=False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
    else:  # different degree of supervision (e.g., COCO-supervision)
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

        # Divide into labeled and unlabeled sets according to supervision percentage
        label_dicts, unlabel_dicts = divide_label_unlabel(
            dataset_dicts,
            cfg.DATALOADER.SUP_PERCENT,
            cfg.DATALOADER.RANDOM_DATA_SEED,
            cfg.DATALOADER.RANDOM_DATA_SEED_PATH,
        )

    label_dataset = DatasetFromList(label_dicts, copy=False)
    # exclude the labeled set from unlabeled dataset
    unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)
    # include the labeled set in unlabel dataset
    # unlabel_dataset = DatasetFromList(dataset_dicts, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    label_dataset = MapDataset(label_dataset, mapper)
    unlabel_dataset = MapDataset(unlabel_dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        label_sampler = TrainingSampler(len(label_dataset))
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return build_semisup_batch_data_loader_two_crop(
        (label_dataset, unlabel_dataset),
        (label_sampler, unlabel_sampler),
        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# batch data loader
def build_semisup_batch_data_loader_two_crop(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    label_dataset, unlabel_dataset = dataset
    label_sampler, unlabel_sampler = sampler

    if aspect_ratio_grouping:
        label_data_loader = torch.utils.data.DataLoader(
            label_dataset,
            sampler=label_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        unlabel_data_loader = torch.utils.data.DataLoader(
            unlabel_dataset,
            sampler=unlabel_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedSemiSupDatasetTwoCrop(
            (label_data_loader, unlabel_data_loader),
            (batch_size_label, batch_size_unlabel),
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")
