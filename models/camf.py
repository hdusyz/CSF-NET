from torch import nn, einsum
import os
import copy
import torch
from einops import rearrange
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class CMFA(nn.Module):
    def __init__(self,img_dim,tab_dim,hid_dim,heads=4,dropout=0.2):
        super().__init__()

        self.fi1=nn.Linear(img_dim,hid_dim)
        self.fi2=nn.Linear(hid_dim,hid_dim)
        self.ft1=nn.Linear(tab_dim,hid_dim)
        self.ft2=nn.Linear(hid_dim,hid_dim)

        self.conv_i1 = nn.Linear(hid_dim, hid_dim)
        self.conv_i2 = nn.Linear(hid_dim, hid_dim)
        self.conv_i3 = nn.Linear(hid_dim, hid_dim)
        self.conv_t1 = nn.Linear(hid_dim, hid_dim)
        self.conv_t2 = nn.Linear(hid_dim, hid_dim)
        self.conv_t3 = nn.Linear(hid_dim, hid_dim)

        self.self_attn_V = MultiheadAttention(hid_dim, heads,dropout=dropout)
        self.self_attn_T = MultiheadAttention(hid_dim, heads,dropout=dropout)
        
    def forward(self,i,t):
        #residual_i = i

        i_ = self.fi1(i)
        i_=F.relu(i_)
        t_ = self.ft1(t)
        t_=F.relu(t_)
        residual_i_ = i_
        residual_t_ = t_

        v1 = F.relu(self.conv_i1(i_))
        k1 = F.relu(self.conv_i2(i_))
        q1 = F.relu(self.conv_i3(i_))
        v2 = F.relu(self.conv_t1(t_))
        k2 = F.relu(self.conv_t2(t_))
        q2 = F.relu(self.conv_t3(t_))

        V_ = self.self_attn_V(q2, k1, v1)[0]
        T_ = self.self_attn_T(q1, k2, v2)[0]
        V_ = V_ + residual_i_
        T_ = T_ + residual_t_

        V_ = self.fi2(V_)
        T_ = self.ft2(T_)

        #V_ = V_ + residual_i    

        return torch.cat((V_,T_),1) 