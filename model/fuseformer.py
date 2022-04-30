''' Fuseformer for Video Inpainting
'''
import numpy as np
import time
import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.spectral_norm import spectral_norm as _spectral_norm
from itertools import product
from core.utils import To_ndim
from core.dist import get_local_rank

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.group = [1, 2, 4, 8, 1]
        self.layers = nn.ModuleList([
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 512, kernel_size=3, stride=1, padding=1, groups=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(768, 384, kernel_size=3, stride=1, padding=1, groups=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(640, 256, kernel_size=3, stride=1, padding=1, groups=8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, groups=1),
            nn.LeakyReLU(0.2, inplace=True)
        ])

    def forward(self, x):
        bt, c, h, w = x.size()
        h, w = h//4, w//4
        out = x
        for i, layer in enumerate(self.layers):
            if i == 8:
                x0 = out
            if i > 8 and i % 2 == 0:
                g = self.group[(i - 8) // 2]
                x = x0.view(bt, g, -1, h, w)
                o = out.view(bt, g, -1, h, w)
                out = torch.cat([x, o], 2).view(bt, -1, h, w)
            out = layer(out)

        return out


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        channel = 256
        hidden = 512
        stack_num = 3
        stack_num_swap = 2  # To change the transformer blocks that are being stuck
        num_head = 4
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (60, 108)
        blocks = []

        dropout = 0.
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)

        local_rank = get_local_rank()

        if torch.cuda.is_available():
            device = torch.device("cuda:{}".format(local_rank))
        else:
            device = 'cpu'

        for _ in range(stack_num):  # 3
            blocks.append(TransformerBlock(hidden=hidden, num_head=num_head, dropout=dropout, n_vecs=n_vecs,
                                           t2t_params=t2t_params, device = device))
        for _ in range(stack_num_swap):  # 2
            blocks.append(TransformerBlockSWAP(hidden=hidden, num_head=num_head, dropout=dropout, n_vecs=n_vecs,
                                               t2t_params=t2t_params, device = device))
        for _ in range(stack_num):  # 3
            blocks.append(TransformerBlock(hidden=hidden, num_head=num_head, dropout=dropout, n_vecs=n_vecs,
                                           t2t_params=t2t_params, device = device))
        self.transformer = blocks
        self.ss = SoftSplit(channel // 2, hidden, kernel_size, stride, padding, dropout=dropout)
        self.add_pos_emb = AddPosEmb(n_vecs, hidden)
        self.sc = SoftComp(channel // 2, hidden, output_size, kernel_size, stride, padding, device=device)
        self.stack_num = stack_num
        self.stack_num_swap = stack_num_swap

        self.encoder = Encoder()

        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconv(channel // 2, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames, masked_semantic_maps, masks):
        # extracting features
        b, t, c, h, w = masked_frames.size()  # [2, 5, 3, 240, 432]
        time0 = time.time()

        enc_feat = self.encoder(masked_frames.view(b * t, c, h, w))  # [10, 128, 60, 108]
        with torch.no_grad():
            enc_feat_sem = self.encoder(masked_semantic_maps.view(b * t, c, h, w))

        trans_feat = self.ss(enc_feat, b)
        trans_feat = self.add_pos_emb(trans_feat)

        trans_feat_sem = self.ss(enc_feat_sem, b)
        trans_feat_sem = self.add_pos_emb(trans_feat_sem)  # [2, 3600, 512]

        sem_results = []
        img_results = []

        for block in self.transformer[0:3]:  # 0, 1, 2
            trans_feat, trans_feat_sem = block(trans_feat, trans_feat_sem)

        for block in [self.transformer[3]]:  # 3
            trans_comp, trans_comp_sem = block(enc_feat_sem,
                                               enc_feat,
                                               trans_feat,
                                               trans_feat_sem,
                                               t, masks)
        sem_results.append(trans_comp_sem)
        img_results.append(trans_comp)

        with torch.no_grad():
            enc_feat = self.encoder(trans_comp.view(b * t, c, h, w))
            trans_feat = self.ss(enc_feat, b)
            trans_feat = self.add_pos_emb(trans_feat)

        for block in [self.transformer[4]]:    # 4
            trans_comp, trans_comp_sem = block(enc_feat_sem,
                                               enc_feat,
                                               trans_feat,
                                               trans_feat_sem,
                                               t, masks)
        sem_results.append(trans_comp_sem)
        img_results.append(trans_comp)

        with torch.no_grad():
            enc_feat = self.encoder(trans_comp.view(b * t, c, h, w))
            trans_feat = self.ss(enc_feat, b)
            trans_feat = self.add_pos_emb(trans_feat)

        for block in self.transformer[5:8]:
            trans_feat, trans_feat_sem = block(trans_feat, trans_feat_sem)

        trans_feat = self.sc(trans_feat, t)
        enc_feat = enc_feat + trans_feat
        output = self.decoder(enc_feat)
        output = torch.tanh(output)

        return output, sem_results


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class deconvSWAP(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0, scale_factor=1, mode='bilinear', device=None):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding, device=device)

    def forward(self, x):
        if self.mode == 'nearest':
            x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
            return self.conv(x)

        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,  align_corners=True)
        return self.conv(x)

# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value, m=None):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        if m is not None:
            scores.masked_fill_(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class AddPosEmb(nn.Module):
    def __init__(self, n, c):
        super(AddPosEmb, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, n, c).float().normal_(mean=0, std=0.02), requires_grad=True)
        self.num_vecs = n

    def forward(self, x):
        b, n, c = x.size()
        x = x.view(b, -1, self.num_vecs, c)
        x = x + self.pos_emb
        x = x.view(b, n, c)
        return x


class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding, dropout=0.1):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, b):
        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat)
        feat = feat.view(b, -1, feat.size(2))
        feat = self.dropout(feat)
        return feat


class SoftComp(nn.Module):
    def __init__(self, channel, hidden, output_size, kernel_size, stride, padding, device=None):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out, device=device)
        self.t2t = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        h, w = output_size
        self.device = device
        self.bias = nn.Parameter(torch.zeros((channel, h, w), dtype=torch.float32), requires_grad=True)

    def forward(self, x, t):
        feat = self.embedding(x)
        b, n, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = self.t2t(feat) + self.bias[None].to(self.device)

        return feat


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, head, p=0.1, device = None):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model,  device = device)
        self.value_embedding = nn.Linear(d_model, d_model,  device = device)
        self.key_embedding = nn.Linear(d_model, d_model,  device = device)
        self.output_linear = nn.Linear(d_model, d_model,  device = device)
        self.attention = Attention(p=p)
        self.head = head

    def forward(self, x):
        b, n, c = x.size()
        c_h = c // self.head
        key = self.key_embedding(x)
        key = key.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        query = self.query_embedding(x)
        query = query.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        value = self.value_embedding(x)
        value = value.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        att, _ = self.attention(query, key, value)
        att = att.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        output = self.output_linear(att)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x


class FusionFeedForward(nn.Module):
    def __init__(self, d_model, p=0.1, n_vecs=None, t2t_params=None, device = None):
        super(FusionFeedForward, self).__init__()
        # We set d_ff as a default to 1960
        hd = 1960
        self.conv1 = nn.Sequential(
            nn.Linear(d_model, hd, device = device))
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(hd, d_model, device = device),
            nn.Dropout(p=p))
        assert t2t_params is not None and n_vecs is not None
        tp = t2t_params.copy()
        self.fold = nn.Fold(**tp)
        del tp['output_size']
        self.unfold = nn.Unfold(**tp)
        self.n_vecs = n_vecs

    def forward(self, x):
        x = self.conv1(x)
        b, n, c = x.size()
        normalizer = x.new_ones(b, n, 49).view(-1, self.n_vecs, 49).permute(0, 2, 1)
        x = self.unfold(self.fold(x.view(-1, self.n_vecs, c).permute(0, 2, 1)) / self.fold(normalizer)).permute(0, 2,
                                                                                                                1).contiguous().view(
            b, n, c)
        x = self.conv2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, num_head=4, dropout=0.1, n_vecs=None, t2t_params=None, device = None):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout, device = device)
        self.ffn = FusionFeedForward(hidden, p=dropout, n_vecs=n_vecs, t2t_params=t2t_params, device = device)
        self.norm1 = nn.LayerNorm(hidden, device = device)
        self.norm2 = nn.LayerNorm(hidden, device = device)
        self.dropout = nn.Dropout(p=dropout)

        self.attention_sem = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout, device = device)
        self.ffn_sem = FusionFeedForward(hidden, p=dropout, n_vecs=n_vecs, t2t_params=t2t_params, device = device)
        self.norm1_sem = nn.LayerNorm(hidden, device = device)
        self.norm2_sem = nn.LayerNorm(hidden, device = device)
        self.dropout_sem = nn.Dropout(p=dropout)

    def forward(self, input, input_sem):
        x = self.norm1(input)
        x = input + self.dropout(self.attention(x))
        y = self.norm2(x)
        x = x + self.ffn(y)

        x_sem = self.norm1_sem(input_sem)
        x_sem = input_sem + self.dropout_sem(self.attention(x_sem))
        y_sem = self.norm2_sem(x_sem)
        x_sem = x_sem + self.ffn_sem(y_sem)

        return x, x_sem



class TransformerBlockSWAP(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    def __init__(self, hidden=128, num_head=4, dropout=0.1, n_vecs=None, t2t_params=None,
                 channel=256, kernel_size=(7, 7), padding=(3, 3), stride=(3, 3), output_size=(60, 108), device=None):
        super().__init__()

        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout, device=device)
        self.ffn = FusionFeedForward(hidden, p=dropout, n_vecs=n_vecs, t2t_params=t2t_params, device=device)
        self.norm1 = nn.LayerNorm(hidden, device=device)
        self.norm2 = nn.LayerNorm(hidden, device=device)
        self.dropout = nn.Dropout(p=dropout)

        self.attention_sem = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout, device=device)
        self.ffn_sem = FusionFeedForward(hidden, p=dropout, n_vecs=n_vecs, t2t_params=t2t_params, device=device)
        self.norm1_sem = nn.LayerNorm(hidden, device=device)
        self.norm2_sem = nn.LayerNorm(hidden, device=device)
        self.dropout_sem = nn.Dropout(p=dropout)

        # decoder: decode frames from features
        self.decoder = nn.Sequential(
            deconvSWAP(channel // 2, 128, kernel_size=3, padding=1, device=device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, device=device),
            nn.LeakyReLU(0.2, inplace=True),
            deconvSWAP(64, 64, kernel_size=3, padding=1, device=device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, device=device)
        )

        self.decoder_sem = nn.Sequential(
            deconvSWAP(channel // 2, 128, kernel_size=3, padding=1, mode='nearest', device=device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, device=device),
            nn.LeakyReLU(0.2, inplace=True),
            deconvSWAP(64, 64, kernel_size=3, padding=1, mode='nearest', device=device),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, device=device)
        )

        self.decoder_to_final_size = nn.Sequential(
            deconvSWAP(3, 3, kernel_size=3, padding=1, scale_factor=2, device=device),
            nn.LeakyReLU(0.2, inplace=True),
            deconvSWAP(3, 3, kernel_size=3, padding=1, scale_factor=2, device=device),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.decoder_to_final_size_sem = nn.Sequential(
            deconvSWAP(3, 3, kernel_size=3, padding=1, scale_factor=2, mode='nearest', device=device),
            nn.LeakyReLU(0.2, inplace=True),
            deconvSWAP(3, 3, kernel_size=3, padding=1, scale_factor=2, mode='nearest', device=device),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.add_pos_emb = AddPosEmb(n_vecs, hidden)
        self.add_pos_emb_sem = AddPosEmb(n_vecs, hidden)
        self.sc = SoftComp(channel // 2, hidden, output_size, kernel_size, stride, padding, device=device)
        self.encoder = Encoder()
        self.to_ndim = To_ndim(device=device)
        self.t2t = torch.nn.Fold(output_size=output_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.swap = SWAP(device=device)
        self.output_size = output_size

    def forward(self, enc_feat_sem, enc_feat, input, input_sem, t, masks):
        x_feat = self.norm1(input)                          # [2, 3600, 512]
        x_feat = input + self.dropout(self.attention(x_feat))
        y_feat = self.norm2(x_feat)
        x_feat = x_feat + self.ffn(y_feat)                  # [2, 3600, 512]

        x_sem_feat = self.norm1_sem(input_sem)              # [2, 3600, 512]
        x_sem_feat = input_sem + self.dropout_sem(self.attention(x_sem_feat))
        y_sem_feat = self.norm2_sem(x_sem_feat)
        x_sem_feat = x_sem_feat + self.ffn_sem(y_sem_feat)  # [2, 3600, 512]

        x = self.sc(x_feat, t)
        x = enc_feat + x
        x = self.decoder(x)
        x = torch.tanh(x)                 # [10, 3, 60, 108]

        x_sem = self.sc(x_sem_feat, t)
        x_sem = enc_feat_sem + x_sem
        x_sem = self.decoder_sem(x_sem)
        x_sem = torch.tanh(x_sem)         # [bt, 3, w//4, h//4]

        x_sem_ndim = self.to_ndim(x_sem.permute(0, 3, 2, 1)).permute(0, 2, 1, 3)  # [bt, 3, w//4, h//4] -> [bt, h//4, w//4, N])

        x = self.swap(x, x_sem_ndim, masks)  # [bt, 3,  60, 108]
        x = self.decoder_to_final_size(x)    # [bt, 3, 240, 432]

        x_sem = self.decoder_to_final_size_sem(x_sem)  # [20, 3, 60, 108] -> [20, 3, 240, 432]

        x_sem = self.to_ndim(x_sem.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).float()  # [20, 133, 240, 432]

        return x, x_sem


# ############################ SWAP BLOCK ################################

class SWAP(nn.Module):
    def __init__(self, k=12, stride=(12, 12), padding=(0, 0), N=133, device = None):
        super(SWAP, self).__init__()
        self.k = k
        self.N = N
        self.stride = stride
        self.padding = padding
        self.unfold = nn.Unfold(kernel_size=(k, k), stride=stride, padding=padding)
        self.fold = nn.Fold(output_size=(60, 108), kernel_size=(k, k), stride=stride, padding=padding)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device


    def forward(self, I, S, M):
        orig_I, orig_S, orig_M = I.float(), S.float(), M.float()
        I, S, M = orig_I.unsqueeze(-1), orig_S.unsqueeze(-1).permute(0, 4, 1, 2, 3), orig_M.unsqueeze(-1)
        bt, N = I.shape[0], self.N
        h, w = 240, 432

        U = I * S * M
        K = I * S * (1 - M)

        K = K.permute(0, 4, 1, 2, 3).reshape(bt * N, 3, h // 4, w // 4)
        U = U.permute(0, 4, 1, 2, 3).reshape(bt * N, 3, h // 4, w // 4)

        K = self.unfold(K)
        U = self.unfold(U)
        _, _, p = K.shape

        W = torch.bmm(torch.transpose(U, 1, 2), K)

        W = F.normalize(W, p=1, dim=-1)

        W = W.unsqueeze(-1).permute(0, 3, 1, 2)
        K = K.unsqueeze(-2)
        WK = torch.sum(W*K, -1)

        output = self.fold(WK)
        output = output.view(bt, N, 3, h // 4, w // 4).sum(1).to(self.device)
        output = orig_I * (1 - orig_M) + output * orig_M

        return output

# ######################################################################
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 32

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels=in_channels, out_channels=nf * 1, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 1, nf * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 2, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                    padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(nf * 4, nf * 4, kernel_size=(3, 5, 5),
                      stride=(1, 2, 2), padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        # T, C, H, W = xs.shape
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)  # B, C, T, H, W
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)  # B, T, C, H, W
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
