import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.RevIN import RevIN
from models.decomposition import Decomposition
from models.KAN import KANLinear  # 导入 KANLinear 类

class WPKANCore(nn.Module):
    """
    WPKAN Core model. This class orchestrates the decomposition,
    token mixing, and embedding mixing.
    """
    def __init__(self, 
                 input_length = [], 
                 pred_length = [],
                 wavelet_name = [],
                 level = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 dropout = [],
                 embedding_dropout = [],
                 tfactor = [],
                 dfactor = [],
                 device = [],
                 patch_len = [],
                 patch_stride = [],
                 no_decomposition = [],
                 use_amp = []):
        
        super(WPKANCore, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition 
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        
        self.Decomposition_model = Decomposition(input_length = self.input_length, 
                                        pred_length = self.pred_length,
                                        wavelet_name = self.wavelet_name,
                                        level = self.level,
                                        batch_size = self.batch_size,
                                        channel = self.channel,
                                        d_model = self.d_model,
                                        tfactor = self.tfactor,
                                        dfactor = self.dfactor,
                                        device = self.device,
                                        no_decomposition = self.no_decomposition,
                                        use_amp = self.use_amp)
        
        self.input_w_dim = self.Decomposition_model.input_w_dim # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim # list of the length of the predicted coefficient series

        # (m+1) number of resolutionBranch
        self.resolutionBranch = nn.ModuleList([ResolutionBranch(input_seq = self.input_w_dim[i],
                                                pred_seq = self.pred_w_dim[i],
                                                batch_size = self.batch_size,
                                                channel = self.channel,
                                                d_model = self.d_model,
                                                dropout = self.dropout,
                                                embedding_dropout = self.embedding_dropout,
                                                tfactor = self.tfactor,
                                                dfactor = self.dfactor,
                                                patch_len = self.patch_len,
                                                patch_stride = self.patch_stride) for i in range(len(self.input_w_dim))])
        
        self.revin = RevIN(self.channel, eps=1e-5, affine = True, subtract_last = False)
        
    def forward(self, xL):
        '''
        Parameters
        ----------
        xL : Look back window: [Batch, look_back_length, channel]

        Returns
        -------
        xT : Prediction time series: [Batch, prediction_length, output_channel]
        '''
        
        x = self.revin(xL, 'norm')
        x = x.transpose(1, 2) # [batch, channel, look_back_length]
        
        # xA: approximation coefficient series, 
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series
        
        xA, xD = self.Decomposition_model.transform(x) 
        
        yA = self.resolutionBranch[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)
        
        y = self.Decomposition_model.inv_transform(yA, yD) 
        y = y.transpose(1, 2)
        y = y[:, -self.pred_length:, :] # decomposition output is always even, but pred length can be odd
        xT = self.revin(y, 'denorm')
        
        return xT


class ResolutionBranch(nn.Module):
    def __init__(self, 
                 input_seq = [],
                 pred_seq = [],
                 batch_size = [],
                 channel = [],
                 d_model = [],
                 dropout = [],
                 embedding_dropout = [],
                 tfactor = [], 
                 dfactor = [],
                 patch_len = [],
                 patch_stride = []):
        super(ResolutionBranch, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len 
        self.patch_stride = patch_stride 
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)
        
        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model) # shared among all channels
        
        # 实例化了两个 KAN-based Mixer 块
        self.mixer1 = Mixer(input_seq = self.patch_num, 
                            d_model = self.d_model,
                            dropout = self.dropout,
                            tfactor = self.tfactor, 
                            dfactor = self.dfactor)
        self.mixer2 = Mixer(input_seq = self.patch_num, 
                            d_model = self.d_model,
                            dropout = self.dropout,
                            tfactor = self.tfactor, 
                            dfactor = self.dfactor)
                            
        self.norm = nn.BatchNorm2d(self.channel)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout) 
        self.head = nn.Sequential(nn.Flatten(start_dim = -2 , end_dim = -1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_seq))
        self.revin = RevIN(self.channel)
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : input coefficient series: [Batch, channel, length_of_coefficient_series]
        
        Returns
        -------
        out : predicted coefficient series: [Batch, channel, length_of_pred_coeff_series]
        '''
        
        x = x.transpose(1, 2)
        x = self.revin(x, 'norm')
        x = x.transpose(1, 2)
        
        x_patch = self.do_patching(x) 
        x_patch  = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch)) 
        
        out =  self.mixer1(x_emb) 
        res = out
        out = res + self.mixer2(out) 
        out = self.norm(out) 
        
        out = self.head(out) 
        out = out.transpose(1, 2)
        out = self.revin(out, 'denorm')
        out = out.transpose(1, 2)
        return out
    
    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim = -1)
        x_patch = x_new.unfold(dimension = -1, size = self.patch_len, step = self.patch_stride) 
        return x_patch 
        
        
class Mixer(nn.Module):
    
    # 基于模板文件架构的 KAN Mixer 块
    
    def __init__(self, 
                 input_seq = [],
                 d_model = [],
                 dropout = [],
                 tfactor = [],
                 dfactor = []):
        super(Mixer, self).__init__()
        self.input_seq = input_seq
        self.d_model = d_model
        self.dropout = dropout
        self.tfactor = tfactor # expansion factor for patch mixer
        self.dfactor = dfactor # expansion factor for embedding mixer
        
        self.tMixer = TokenMixer(input_seq = self.input_seq, pred_seq = self.input_seq, dropout = self.dropout, factor = self.tfactor)
        self.embeddingMixer = EmbeddingMixer(d_model=self.d_model, dfactor=self.dfactor, dropout=self.dropout)

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.norm1 = nn.BatchNorm2d(d_model) # Norm on d_model dimension
        self.norm2 = nn.BatchNorm2d(d_model) # Norm on d_model dimension
        
    def forward(self, x):
        '''
        Parameters
        ----------
        x : input: [Batch, Channel, Patch_number, d_model]

        Returns
        -------
        x: output: [Batch, Channel, Patch_number, d_model]
        '''
        # Token Mixing part
        x_res = x
        x = self.norm1(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # Norm across d_model
        x = x.permute(0, 3, 1, 2) # [B, d_model, C, patch_num]
        x = self.tMixer(x.contiguous())
        x = x.permute(0, 2, 3, 1) # [B, C, patch_num, d_model]
        x = x_res + self.dropoutLayer(x)

        # Embedding Mixing part
        x_res = x
        x = self.norm2(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) # Norm across d_model
        x = self.embeddingMixer(x.contiguous())
        x = x_res + self.dropoutLayer(x)
        return x 
    
class TokenMixer(nn.Module):
    """
    Token Mixer class, now using KANLinear.
    It mixes information across the patch dimension.
    """
    def __init__(self, input_seq=[], pred_seq=[], dropout=[], factor=[]):
        super(TokenMixer, self).__init__()
        self.mixer1 = KANLinear(
            in_features=input_seq,
            out_features=pred_seq * factor,
            grid_size=3,
            spline_order=3,
        )
        self.mixer2 = KANLinear(
            in_features=pred_seq * factor,
            out_features=pred_seq,
            grid_size=3,
            spline_order=3,
        )
        self.dropoutLayer = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input x: [Batch, d_model, Channel, Patch_number]
        Output y: [Batch, d_model, Channel, pred_seq]
        """
        x = self.mixer1(x)
        x = F.gelu(x)
        x = self.dropoutLayer(x)
        x = self.mixer2(x)
        return x

class EmbeddingMixer(nn.Module):
    """
    Embedding Mixer class, now using KANLinear.
    It mixes information across the embedding (d_model) dimension.
    """
    def __init__(self, d_model=[], dfactor=[], dropout=[]):
        super(EmbeddingMixer, self).__init__()
        self.mixer1 = KANLinear(
            in_features=d_model,
            out_features=d_model * dfactor,
            grid_size=3, 
            spline_order=3,
        )
        self.mixer2 = KANLinear(
            in_features=d_model * dfactor,
            out_features=d_model,
            grid_size=3,
            spline_order=3,
        )
        self.dropoutLayer = nn.Dropout(dropout)

    def forward(self, x):
        """
        Input x: [Batch, Channel, Patch_number, d_model]
        Output y: [Batch, Channel, Patch_number, d_model]
        """
        x = self.mixer1(x)
        x = F.gelu(x)
        x = self.dropoutLayer(x)
        x = self.mixer2(x)
        return x