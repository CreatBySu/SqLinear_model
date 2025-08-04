import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearMlp(nn.Module):
    """Simplified MLP for linear block"""
    def __init__(self, in_features, hidden_features, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class HLI(nn.Module):
    """Hierarchical Linear Interaction Block"""
    def __init__(self, hidden_size, num, size, rank, mlp_ratio=4.0):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size

        # Inter-patch interaction components
        self.inter_patch_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.inter_patch_interaction = LinearMlp(hidden_size, hidden_size)
        self.inter_patch_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.inter_patch_mlp = LinearMlp(hidden_size, mlp_hidden_dim)

        # Intra-patch interaction components
        self.intra_patch_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.intra_patch_interaction = LinearMlp(hidden_size, hidden_size)
        self.intra_patch_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.intra_patch_mlp = LinearMlp(hidden_size, mlp_hidden_dim)

        self.low_proj1 = nn.Linear(num, rank, bias=False)
        self.low_proj2 = nn.Linear(rank, num, bias=False)

    def forward(self, x):
        # Input: x [B, T, P*N, D] -> [B, T, P, N, D]
        B, T, _, D = x.shape
        P, N = self.num, self.size  # P: patch num, N: patch size
        assert self.num * self.size == _
        x = x.reshape(B, T, P, N, D)

        # Inter-patch interaction: [B, T, P, N, D] -> [B*T*N, P, D]
        x_inter_patch = x.transpose(2, 3).reshape(B*T*N, P, D)  # (B*T*N, P, D)
        x_inter_patch_norm = self.inter_patch_norm(x_inter_patch)
        inter_patch_out = self.inter_patch_interaction(x_inter_patch_norm)  # (B*T*N, P, D)
        
        # Projection: [B*T*N, P, D] -> [B*T*N, D, P] -> [B*T*N, P, D]
        inter_patch_out_t = inter_patch_out.transpose(1, 2)  # (B*T*N, D, P)
        inter_patch_out_proj = self.low_proj1(inter_patch_out_t)  
        inter_patch_out_proj = self.low_proj2(inter_patch_out_proj)  
        inter_patch_out = inter_patch_out_proj.transpose(1, 2)  # (B*T*N, P, D)
        
        # Reshape back: [B*T*N, P, D] -> [B, T, P, N, D]
        x = x + inter_patch_out.reshape(B, T, N, P, D).transpose(2, 3)
        x = x + self.inter_patch_mlp(self.inter_patch_norm2(x))
        
        # Intra-patch interaction: [B, T, P, N, D] -> [B*T*P, N, D]
        x_intra_patch = x.reshape(B*T*P, N, D)  # (B*T*P, N, D)
        x_intra_patch_norm = self.intra_patch_norm(x_intra_patch)
        intra_patch_out = self.intra_patch_interaction(x_intra_patch_norm)  # (B*T*P, N, D)
        x = x + intra_patch_out.reshape(B, T, P, N, D)  # [B*T*P, N, D] -> [B, T, P, N, D]
        x = x + self.intra_patch_mlp(self.intra_patch_norm2(x))
         
        # Output: [B, T, P, N, D] -> [B, T, P*N, D]
        return x.reshape(B, T, -1, D)

class SqLinear(nn.Module):
    """Spatial-temporal traffic forecasting model with hierarchical linear interactions"""
    def __init__(self, tem_patchsize, tem_patchnum,
                        node_num, spa_patchsize, spa_patchnum,
                        tod, dow,
                        layers,
                        input_dims, node_dims, tod_dims, dow_dims,
                        ori_parts_idx, reo_parts_idx, reo_all_idx,
                        rank=16,
                        mlp_ratio=1.0
                ):
        super(SqLinear, self).__init__()
        self.node_num = node_num
        self.ori_parts_idx, self.reo_parts_idx = ori_parts_idx, reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.tod, self.dow = tod, dow

        # Model dimensions
        dims = input_dims + tod_dims + dow_dims + node_dims

        # Spatio-temporal embedding layers
        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims, 
                                   kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)
        self.node_emb = nn.Parameter(torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # Hierarchical linear interaction encoder
        self.spa_encoder = nn.ModuleList([
            HLI(dims, spa_patchnum, spa_patchsize, rank, mlp_ratio=mlp_ratio) 
            for _ in range(layers)
        ])
        # Projection decoder
        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum*dims, 
                                       out_channels=tem_patchsize*tem_patchnum, 
                                       kernel_size=(1, 1), bias=True)

    def forward(self, x, te):
        # x: [B,T,N,1] input traffic
        # te: [B,T,N,2] time information

        # Spatio-temporal embedding
        embeded_x = self.embedding(x, te)
        rex = embeded_x[:,:,self.reo_all_idx,:] # select patched points

        # Hierarchical linear interaction encoding
        for block in self.spa_encoder:
            rex = block(rex)

        # Restore original spatial structure
        orginal = torch.zeros(rex.shape[0],rex.shape[1],self.node_num,rex.shape[-1]).to(x.device)
        orginal[:,:,self.ori_parts_idx,:] = rex[:,:,self.reo_parts_idx,:] # back to the original indices

        # Projection decoding
        pred_y = self.regression_conv(orginal.transpose(2,3).reshape(orginal.shape[0],-1,orginal.shape[-2],1))

        return pred_y # [B,T,N,1]

    def embedding(self, x, te):
        b,t,n,_ = x.shape

        # Combine input traffic with time features
        x1 = torch.cat([x,(te[...,0:1]/self.tod),(te[...,1:2]/self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1,3)).transpose(1,3)
        t, d = input_data.shape[1], input_data.shape[-1]        

        # Add time of day embedding
        t_i_d_data = te[:, -input_data.shape[1]:, :, 0]
        input_data = torch.cat([input_data, self.time_in_day_emb[(t_i_d_data).type(torch.LongTensor)]], -1)

        # Add day of week embedding
        d_i_w_data = te[:, -input_data.shape[1]:, :, 1]
        input_data = torch.cat([input_data, self.day_in_week_emb[(d_i_w_data).type(torch.LongTensor)]], -1)

        # Add spatial embedding
        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)

        return input_data
    
    def get_n_param(self):
        """Count trainable parameters"""
        n_param = 0
        for param in self.parameters():
            if param.requires_grad:
                n_param += torch.numel(param)
        return n_param
