import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrUpBlock
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch import Tensor


class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:

        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)  

    def forward(self, x):

        #  output = word_embedding + positional_embedding
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]

class GuideDecoderLayer(nn.Module):

    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=24, embed_dim:int=768):

        super(GuideDecoderLayer, self).__init__()

        self.in_channels = in_channels

        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)

        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=1,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)

        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim,in_channels),
            nn.LeakyReLU(),
        )

        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels,max_len=output_text_len)

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)

        self.scale = nn.Parameter(torch.tensor(0.01),requires_grad=True)


    def forward(self,x,txt):

        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        txt = self.text_project(txt)

        # Self-Attention
        vis2 = self.norm1(x)
        q = k = self.vis_pos(vis2)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = x + vis2

        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2,_ = self.cross_attn(query=self.vis_pos(vis2),
                                   key=self.txt_pos(txt),
                                   value=txt)
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.scale*vis2

        return vis

class GuideDecoder(nn.Module):

    def __init__(self,in_channels, out_channels, spatial_size, text_len) -> None:

        super().__init__()

        self.guide_layer = GuideDecoderLayer(in_channels,text_len)   # for skip
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2,in_channels,out_channels,3,2,norm_name='BATCH')

    
    def forward(self, vis, skip_vis, txt):

        if txt is not None:
            vis =  self.guide_layer(vis, txt)

        vis = rearrange(vis,'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size)
        skip_vis = rearrange(skip_vis,'B (H W) C -> B C H W',H=self.spatial_size*2,W=self.spatial_size*2)

        output = self.decoder(vis,skip_vis)
        output = rearrange(output,'B C H W -> B (H W) C')

        return output

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

def deconv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))

def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))
    
class Bridger(nn.Module):
    def __init__(self,
                 d_img = 512,
                 d_txt = 768,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = 2,
                 num_layers = 12,
                 stage_id = 1
                ):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.stage_id = stage_id

        self.fusion_v = Interactor(d_model=d_model, nhead=nhead)
        self.fusion_t = Interactor(d_model=d_model, nhead=nhead)
        self.zoom_in = nn.Conv2d(d_img, d_model, kernel_size=strides, stride=strides, bias=False)
        if self.stage_id == 4:
            self.zoom_out = nn.ConvTranspose2d(d_model, d_img, kernel_size=strides, stride=strides, 
                                                padding=0, output_padding=1, bias=False)
        else:
            self.zoom_out = nn.ConvTranspose2d(d_model, d_img, kernel_size=strides, stride=strides, bias=False)
        self.linear1 = nn.Linear(d_txt, d_model)
        self.linear2 = nn.Linear(d_model, d_txt)
        self.ln_v = nn.LayerNorm(d_model)
        self.ln_t = nn.LayerNorm(d_model)

        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, vis, txt):
        # residual operation
        txt = txt.permute(1, 0, 2)  # NLD -> LND
        v = vis.clone()
        t = txt.clone()
        t = t.float()
        last_v, last_t = v, t
        # dimension reduction
        v = self.zoom_in(v)
        t = self.linear1(t)
        # multi modal fusion
        B, C, H, W = v.shape
        v = v.reshape(B, C, -1).permute(2, 0, 1) # B, C, H, W -> B, C, HW -> HW, B, C(676, 64, 256)
        v, t = self.ln_v(v), self.ln_t(t)
        v, t = self.fusion_v(v, t), self.fusion_t(t, v)
        v = v.permute(1, 2, 0).reshape(B, -1, H, W) # HW, B, C -> B, C, HW -> B, C, H, W
        # dimension recovery
        v = self.zoom_out(v)                
        t = self.linear2(t)
        # residual connect
        vis = vis + v
        txt = txt + t   

        # After fusion
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        # txt = backbone.ln_final(txt).type(backbone.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # state = txt[torch.arange(txt.shape[0]),
                #   text.argmax(dim=-1)] @ backbone.text_projection

        # forward
        return vis, txt


class Interactor(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1,
                 activation="relu", ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout) 

        self.activation = _get_activation_fn(activation)   

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_key_padding_mask: Optional[Tensor] = None,                
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # self attn
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        tgt2 = self.self_attn(q, k, value=v, attn_mask=None,
                              key_padding_mask=tgt_key_padding_mask)[0] # [H*W, B, C]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)      

        # cross attn                
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1), 
                                            nn.BatchNorm2d(out_channels), 
                                            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)
        return x  


class FusedDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, d_img = 512, d_txt = 768, d_model = 64, 
                    nhead = 8, num_stages = 3, strides = 2, num_layers = 12):
        super().__init__()
        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers

        self.fusion_v = Interactor(d_model=d_model, nhead=nhead)
        self.fusion_t = Interactor(d_model=d_model, nhead=nhead)
        self.zoom_in = nn.Conv2d(d_img, d_model, kernel_size=strides, stride=strides, bias=False)
        self.zoom_out = nn.ConvTranspose2d(d_model, d_img, kernel_size=strides, stride=strides, bias=False)
        self.linear1 = nn.Linear(d_txt, d_model)
        self.linear2 = nn.Linear(d_model, d_txt)
        self.ln_v = nn.LayerNorm(d_model)
        self.ln_t = nn.LayerNorm(d_model)

        self.initialize_parameters()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1), 
                                            nn.BatchNorm2d(out_channels), 
                                            nn.ReLU(inplace=True))

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, vis, txt, x2):
        # residual operation
        txt = txt.permute(1, 0, 2)  # NLD -> LND
        v = vis.clone()
        t = txt.clone()
        t = t.float()
        last_v, last_t = v, t
        # dimension reduction
        v = self.zoom_in(v)
        t = self.linear1(t)
        # multi modal fusion
        B, C, H, W = v.shape
        v = v.reshape(B, C, -1).permute(2, 0, 1) # B, C, H, W -> B, C, HW -> HW, B, C(676, 64, 256)
        v, t = self.ln_v(v), self.ln_t(t)
        v, t = self.fusion_v(v, t), self.fusion_t(t, v)
        v = v.permute(1, 2, 0).reshape(B, -1, H, W) # HW, B, C -> B, C, HW -> B, C, H, W
        # dimension recovery
        v = self.zoom_out(v)                
        t = self.linear2(t)
        # residual connect
        vis = vis + v
        txt = txt + t   

        # After fusion
        txt = txt.permute(1, 0, 2)  # LND -> NLD
        # txt = backbone.ln_final(txt).type(backbone.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # state = txt[torch.arange(txt.shape[0]),
                #   text.argmax(dim=-1)] @ backbone.text_projection

        # forward

        x1 = self.up(vis)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_bn_relu(x)

        return x, txt


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

            
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    
class ECA_ChannelAttention(nn.Module):
    def __init__(self, in_channels, k_size=3):
        super(ECA_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x0*y.expand_as(x0)


class Dec_ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Dec_ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x0*self.sigmoid(out)


class Dec_SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(Dec_SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x*self.sigmoid(out)
