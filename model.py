"""SGRAF model"""

import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_

import numpy as np
from collections import OrderedDict

from transformers import BertModel

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """
    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.project = nn.Linear(5, img_dim)
        self.map = nn.Sequential(
            nn.Linear(img_dim, img_dim),
            nn.ReLU(),
            nn.Linear(img_dim, embed_size)
        )

        self.gate_fn = nn.Sequential(
            nn.Linear(img_dim + 5, img_dim),
            nn.ReLU(),
            nn.Linear(img_dim, 1)
        )
        self.node_fn = nn.Sequential(
                nn.Linear(img_dim + 5, img_dim),
                nn.ReLU(),
                nn.Linear(img_dim, img_dim)
            )

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)


    def forward(self, images, bboxes, img_range):
        """Extract image feature vectors."""

        # assuming that the precomputed features are already l2-normalized
        area = (bboxes[:, :, 2] - bboxes[:, :, 0]) * (bboxes[:, :, 3] - bboxes[:, :, 1])
        area = area.unsqueeze(2)
        s_infos = torch.cat([bboxes, area], dim=2)
        k = images.shape[1]
        batch_size = images.shape[0]
        img_dim = images.shape[2]
        n_partial=5

        gate = self.gate_fn(torch.cat([images, s_infos * 0.1], dim=2))
        m = torch.sigmoid(gate) 
        v = self.node_fn(torch.cat([images, s_infos * 0.1], dim=2))

        idx_background = torch.arange(0,k).unsqueeze(1).repeat(1,n_partial)\
                            .unsqueeze(0).repeat(batch_size,1,1).to('cuda:0')
        effe_rels = torch.topk(img_range,n_partial,-1,largest=True)[0]
        topk_idx = torch.topk(img_range,n_partial,-1,largest=True)[1]    
        effe_ids = torch.where(effe_rels==1, topk_idx, idx_background)
        effe_ids_weight = effe_ids.unsqueeze(3)
        effe_ids_feat = effe_ids_weight.repeat(1,1,1,img_dim)

        v_rep = v.unsqueeze(1).repeat(1,k,1,1)
        v_gather = torch.gather(v_rep, dim=2, index=effe_ids_feat) 

        m_raw = m.unsqueeze(1).repeat(1,k,1,1)
        m_gather = torch.gather(m_raw, dim=2, index=effe_ids_weight) 

        out = torch.matmul(m_gather.permute(0, 1, 3, 2), v_gather).squeeze() 
        out = l2norm(out, dim=-1)

        images = images + out

        img_emb = self.map(images) 

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """
    def __init__(self, vocab_size, hidden_size, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(hidden_size, embed_size)


    def forward(self, captions, lengths):
        """Handles variable size captions"""
        # embed word ids to vectors

        bert_attention_mask = (captions != 0).float()

        bert_emb = self.bert(captions, bert_attention_mask)[0]  # B x N x D

        return bert_emb

class word_kv_memory(nn.Module):
    def __init__(self, opt):
        super(word_kv_memory, self).__init__()
        self.opt = opt
        self.no_txtnorm = opt.no_txtnorm
        # self.args = args
        self.word_embedding_a = nn.Embedding(opt.vocab_size, opt.hidden_size)
        self.word_embedding_c = nn.Embedding(opt.feature_size, opt.hidden_size)
        self.fc = nn.Linear(opt.hidden_size * 2, opt.embed_size)
        self.map = nn.Sequential(
            nn.Linear(opt.hidden_size * 2, opt.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(opt.hidden_size * 2, opt.embed_size)
        )

        self.init_weights()

    def init_weights(self):
        self.word_embedding_a.weight.data.uniform_(-0.1, 0.1)
        self.word_embedding_c.weight.data.uniform_(-0.1, 0.1)

        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
    
    def forward(self, captions, hidden_state, targets_feature, feature_mask):

        embedding_a = self.word_embedding_a(captions) 
        embedding_c = self.word_embedding_c(targets_feature) 

        embedding_a = embedding_a.permute(0, 2, 1)
        u = torch.matmul(hidden_state, embedding_a) 

        tmp_word_mask_metrix = torch.clamp(feature_mask, 0, 1)

        exp_u = torch.exp(u)
        delta_exp_u = torch.mul(exp_u, tmp_word_mask_metrix)

        sum_delta_exp_u = torch.sum(delta_exp_u, 2).unsqueeze(-1).repeat(1, 1, delta_exp_u.shape[2])
        p = torch.div(delta_exp_u, sum_delta_exp_u + 1e-10)

        p = p.unsqueeze(-1).repeat(1, 1, 1, hidden_state.shape[2])


        character_seq_len = p.shape[1]
        embedding_c = embedding_c.unsqueeze(1).repeat(1, character_seq_len, 1, 1)

        o = torch.mul(p, embedding_c)
        o = torch.sum(o, 2)
        o = torch.cat((hidden_state, o), 2)
        o = self.fc(o)

        if not self.no_txtnorm:
            o = l2norm(o + 1e-10, dim=-1)

        return o
                
class Sim_vec(nn.Module):

    def __init__(self, embed_dim, sim_dim):
        super(Sim_vec, self).__init__()
        
        self.sim_loc = nn.Linear(embed_dim, sim_dim)

        self.relu = nn.ReLU(inplace=True)
        
        self.sim_eval = nn.Linear(sim_dim, 1)

        self.sim_block = nn.Sequential(
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
        
        self.tanh = nn.Tanh()

        self.n_dim = 16
        
        self.init_weights()

    def forward(self, img_emb, cap_emb, cap_lens, im_start, ca_start, compare_r_f):
        
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)
        embed_size = img_emb.size(2)
        n_block = embed_size // self.n_dim
        
        sim_all = []
        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)     
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # local-global alignment construction
            Context_img, attn = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)

            qry_set = cap_i_expand.view(n_caption, n_word, self.n_dim, n_block)
            ctx_set = Context_img.view(n_caption, n_word, self.n_dim, n_block)

            mvector = cosine_sim(qry_set, ctx_set, dim=-1) 
            
            sim_b = self.sim_block(mvector)

            sim_i = torch.mean(sim_b, dim=1)
            sim_i = self.tanh(sim_i)
            sim_all.append(sim_i)
            
        sim_all = torch.cat(sim_all, 1)
                  
        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """

    queryT = torch.transpose(query, 1, 2)

    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    attn = torch.transpose(attn, 1, 2).contiguous()

    attn = F.softmax(attn*smooth, dim=2)

    attnT = torch.transpose(attn, 1, 2).contiguous()

    contextT = torch.transpose(context, 1, 2)
    weightedContext = torch.bmm(contextT, attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext, attn


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.to('cuda:0')
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class SGRAF(object):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """
    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.vocab_size, opt.hidden_size,
                                   opt.embed_size, opt.num_layers, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm)
        self.sim_vec = Sim_vec(opt.embed_size, opt.sim_dim)
        self.kvmn = word_kv_memory(opt)

        if torch.cuda.is_available():
            self.img_enc.to('cuda:0')
            self.txt_enc.to('cuda:0')
            self.sim_vec.to('cuda:0')
            self.kvmn.to('cuda:0')
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_vec.parameters())
        params += list(self.kvmn.parameters())
        self.params = params

        decay_factor = 1e-4
        all_text_params = list(self.txt_enc.parameters())
        bert_params = list(self.txt_enc.bert.parameters())
        bert_params_ptr = [p.data_ptr() for p in bert_params]
        text_params_no_bert = list()
        for p in all_text_params:
            if p.data_ptr() not in bert_params_ptr:
                text_params_no_bert.append(p)

        self.optimizer = torch.optim.Adam([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 1.0},
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.sim_vec.parameters(), 'lr': opt.learning_rate},
                    {'params': self.kvmn.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate)

        # self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0
        self.data_parallel = False

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()\
                     , self.sim_vec.state_dict(), self.kvmn.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_vec.load_state_dict(state_dict[2])
        self.kvmn.load_state_dict(state_dict[3])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_vec.train()
        self.kvmn.train()
        
    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_vec.eval()
        self.kvmn.eval()

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.sim_vec = nn.DataParallel(self.sim_vec)
        self.kvmn = nn.DataParallel(self.kvmn)
        self.data_parallel = True

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, lengths, bboxes, img_range):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.to('cuda:0')
            captions = captions.to('cuda:0')
            bboxes = bboxes.to('cuda:0')
            img_range = img_range.to('cuda:0')

        # Forward feature encoding
        img_embs = self.img_enc(images, bboxes, img_range)
        bert_embs = self.txt_enc(captions, lengths)
        return img_embs, bert_embs, lengths

    def forward_kvmn(self, captions, cap_embs, targets_feature, feature_mask):
        if torch.cuda.is_available():
            targets_feature = targets_feature.to('cuda:0')
            captions = captions.to('cuda:0')
            feature_mask = feature_mask.to('cuda:0')
        
        cap_feat_embs = self.kvmn(captions, cap_embs, targets_feature, feature_mask)
        return cap_feat_embs

    def forward_sim(self, img_embs, cap_embs, cap_lens, im_start, ca_start, compare_r_f):
            
        sims = self.sim_vec(img_embs, cap_embs, cap_lens, im_start, ca_start, compare_r_f)
        
        return sims

    def forward_loss(self, sims, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Loss', loss.item(), sims.size(0))
        return loss

    def train_emb(self, images, captions, lengths, targets_feature, feature_mask, bboxes, img_range, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_embs, bert_embs, cap_lens = self.forward_emb(images, captions, lengths, bboxes, img_range)
        cap_feat_embs = self.forward_kvmn(captions, bert_embs, targets_feature, feature_mask)
        sims = self.forward_sim(img_embs, cap_feat_embs, cap_lens)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()
