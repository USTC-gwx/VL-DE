"""Data provider"""

import imp
import torch
import torch.utils.data as data

import os
import numpy as np

import json
import pickle
# import nltk


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, tokenizer, vocab, opt):
        self.vocab = vocab
        loc = data_path + '/'
        self.tokenizer = tokenizer

        # load the raw captions
        self.captions = []
        self.captions_feature = []
        self.ranges = None
        self.feature2id = None
        # 
        # /home/gwx/trees/bert/
        with open(opt.dep_path + '%s_precaps.txt' % data_split, 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        with open(opt.dep_path + '%s_cap_features.txt' % data_split, 'r') as f:
            for line in f:
                self.captions_feature.append(line.strip())
        with open(opt.dep_path + '%s_ranges.pickle' % data_split, 'rb') as f:
            self.ranges = pickle.load(f)
        with open(opt.dep_path + 'feature2id.json', 'r') as f:
            self.feature2id = json.load(f)


        # load the image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)

        self.bbox = np.load(loc + '%s_ims_bbx.npy' % data_split)
        self.sizes = np.load(loc + '%s_ims_size.npy' % data_split, allow_pickle=True, encoding='latin1')
        self.img_ranges = np.load(opt.dep_path + '%s_ranges.npy' % data_split)

        # rkiros data has redundancy in images, we divide by 5
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = torch.Tensor(self.images[img_id])

        bboxes = self.bbox[img_id]
        imsize = self.sizes[img_id]
        img_range = self.img_ranges[img_id]

        k = image.shape[0]
        assert k == 36

        for i in range(k):
            bbox = bboxes[i]
            bbox[0] /= imsize['image_w']
            bbox[1] /= imsize['image_h']
            bbox[2] /= imsize['image_w']
            bbox[3] /= imsize['image_h']
            bboxes[i] = bbox

        bboxes = torch.Tensor(bboxes)
        img_range = torch.Tensor(img_range)

        caption = self.captions[index]
        caption_feature = self.captions_feature[index]
        ran = self.ranges[index]

        # convert caption (string) to word ids.
        # cap = []
        # words = caption.lower().split(' ')
        # cap.append(self.vocab('<start>'))
        # #cap = [int(item) for _,item in enumerate(cap)]
        # cap.extend([self.vocab(word) for word in words])
        # cap.append(self.vocab('<end>'))
        # caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        tokens = []
        tokens.append('[CLS]')
        # for i, token in enumerate(caption_tokens):
        #     sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
        #     tokens.extend([sub_token for sub_token in sub_tokens])
        tokens.extend(caption.lower().split(' '))
        tokens.append('[SEP]')

        target = self.tokenizer.convert_tokens_to_ids(tokens)

        # convert caption features (string) to feature ids.
        tokens_feature = caption_feature.lower().split(' ')
        cap_feat = list()
        cap_feat.append(self.feature2id['<start>'])
        for k, token in enumerate(tokens_feature):
            if token in self.feature2id:
                cap_feat.append(self.feature2id[token])
            else:
                cap_feat.append(self.feature2id['<unk>'])
        # cap_feat.extend([self.feature2id[token] for token in tokens])
        cap_feat.append(self.feature2id['<end>'])

        # ran = (np.array(ran) + 1).tolist()
        # ran.insert(0,[i for i in range(len(target))])
        # ran.append([i for i in range(len(target))])

        assert len(target) == len(cap_feat)
        assert len(target) == len(ran)
        caption = torch.Tensor(target)
        caption_feature = torch.Tensor(cap_feat)

        return image, caption, index, img_id, caption_feature, ran, bboxes, img_range

    def __len__(self):
        return self.length


def collate_fn(data):
    """
    Build mini-batch tensors from a list of (image, caption, index, img_id) tuples.
    Args:
        data: list of (image, target, index, img_id) tuple.
            - image: torch tensor of shape (36, 2048).
            - target: torch tensor of shape (?) variable length.
    Returns:
        - images: torch tensor of shape (batch_size, 36, 2048).
        - targets: torch tensor of shape (batch_size, padded_length).
        - lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, captions_feature, ranges, bboxes, img_range = zip(*data)

    # Merge images (convert tuple of 2D tensor to 3D tensor)
    images = torch.stack(images, 0)
    bboxes = torch.stack(bboxes, 0)
    img_range = torch.stack(img_range, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    targets_feature = torch.zeros(len(captions_feature), max(lengths)).long()
    feature_mask = torch.zeros(len(captions_feature), max(lengths), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        targets_feature[i, :end] = captions_feature[i][:end]
        ran = ranges[i] # (len, 2)
        for j, ra in enumerate(ran):
            for k in ra:
                feature_mask[i][j][k] = 1
    
    return images, targets, lengths, targets_feature, feature_mask, bboxes, img_range, ids


def get_precomp_loader(data_path, data_split, tokenizer, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    dset = PrecompDataset(data_path, data_split, tokenizer, vocab, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, vocab, tokenizer, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', tokenizer, vocab, opt,
                                      batch_size, True, workers)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'test', tokenizer, vocab, opt,
                                    100, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, tokenizer, vocab, batch_size, workers, opt):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, tokenizer, vocab, opt,
                                     100, False, workers)
    return test_loader
