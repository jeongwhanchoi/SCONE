import torch
import torch.nn as nn
from . import layers
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_pairwise
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
import sde_lib
import losses_sgm
import sampling_sgm
from ema import ExponentialMovingAverage
import random


get_act = layers.get_act
default_initializer = layers.default_init

class SCONE(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SCONE, self).__init__(conf, training_set, test_set)
        self.config = conf
        self.cl_rate = self.config.lambda_cl
        self.n_layers = self.config.n_layer
        self.T = self.config.T
        self.sde_name = self.config.sde_type
        self.beta_1 = self.config.noise_min
        self.beta_T = self.config.noise_max
        self.lr = self.config.lr
        self.lr_score = self.config.lr_score
        self.weight = self.config.weight
        self.NS = self.config.NS
        self.CL = self.config.CL
        self.save_model = self.config.save_model
        self.sampling_ratio = self.config.sampling_ratio
        score_dim = self.config.score_dim.split(',')
        self.score_dim = [int(num) for num in score_dim]

        self.model = LightGCN(self.data, self.emb_size, self.n_layers)
        self.diff_model = SGM(self.data, self.emb_size, self.score_dim, self.config, act='tanh')

        if self.sde_name.lower() == 'vpsde':
            self.sde = sde_lib.VPSDE(beta_min=self.beta_1, beta_max=self.beta_T, N=self.T)
            self.sampling_eps = 1e-3
        elif self.sde_name.lower()  == 'subvpsde':
            self.sde = sde_lib.subVPSDE(beta_min=self.beta_1, beta_max=self.beta_T, N=self.T)
            self.sampling_eps = 1e-3
        elif self.sde_name.lower()  == 'vesde':
            self.sde = sde_lib.VESDE(sigma_min=self.beta_1, sigma_max=self.beta_T, N=self.T)
            self.sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {self.sde} unknown.")
        

    def train(self):
        model = self.model.cuda()
        diff_model = self.diff_model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        ema_user = ExponentialMovingAverage(diff_model.parameters(), decay=0.999)
        optimizer_score = losses_sgm.get_optimizer(None, diff_model.parameters(), self.lr_score)
        self.state_score = dict(optimizer=optimizer_score, model=diff_model, ema=ema_user, step=0, epoch=0, ml_param=0)
        self.optimize_fn_user = losses_sgm.optimization_manager()
        self.train_step_fn_user = losses_sgm.get_step_fn(self.sde, train=True, optimize_fn=self.optimize_fn_user)
        self.sampling_fn_user = sampling_sgm.get_sampling_fn(None, self.sde, None, None, self.sampling_eps)

        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]

                score_emb = torch.cat([rec_user_emb, rec_item_emb])
                score_loss = self.train_step_fn_user(self.state_score, score_emb)

                optimizer_score.zero_grad()
                score_loss.backward(retain_graph=True)
                self.optimize_fn_user(optimizer_score, diff_model.parameters(), step=self.state_score['step'])
                self.state_score['ema'].update(diff_model.parameters())
                self.state_score['step'] += 1

                if self.NS:
                    cl_loss, negative_embedding_new = self.cal_cl_loss(self.state_score, rec_user_emb, rec_item_emb, [user_idx, pos_idx, neg_idx])
                    rec_loss = bpr_loss(user_emb, pos_item_emb, negative_embedding_new)
                else:
                    cl_loss = self.cal_cl_loss(self.state_score, rec_user_emb, rec_item_emb, [user_idx, pos_idx, neg_idx])
                    rec_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)

                cl_loss = self.cl_rate * cl_loss
                batch_loss =  rec_loss + l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                if n % 100==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'cl_loss', cl_loss.item(), 'diffusion_loss', score_loss.item())

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            self.fast_evaluation(epoch)
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_cl_loss(self, state_score, rec_user_emb,  rec_item_emb, idx):

        if self.CL:
            u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
            i_idx = torch.unique(torch.Tensor(idx[1]+idx[2]).type(torch.long)).cuda()

            embedding_x_0 = torch.cat([rec_user_emb[u_idx], rec_item_emb[i_idx]])
            
            embedding_x_t_list = self.sampling_fn_user(self.state_score['model'], sampling_shape=embedding_x_0.shape, test=embedding_x_0, NS=False, weight=None, sampling_ratio = self.sampling_ratio)
            user_view_1 = embedding_x_t_list[0][:len(u_idx)]
            user_view_2 = embedding_x_t_list[-1][:len(u_idx)]
            item_view_1 = embedding_x_t_list[0][len(u_idx):]
            item_view_2 = embedding_x_t_list[-1][len(u_idx):]

            user_cl_loss = InfoNCE(user_view_1, user_view_2, 0.2)  
            item_cl_loss = InfoNCE(item_view_1, item_view_2, 0.2)
        else:
            user_cl_loss = torch.tensor(0)
            item_cl_loss = torch.tensor(0)


        if self.NS:
            pos_idx = torch.Tensor(idx[1]).type(torch.long)
            neg_idx = torch.Tensor(idx[2]).type(torch.long)
            embedding_x_0 = torch.cat([rec_item_emb[pos_idx], rec_item_emb[neg_idx]])
            negative_embedding_new = self.sampling_fn_user(self.state_score['model'], sampling_shape=embedding_x_0.shape, test=embedding_x_0, NS=True, weight=self.weight, sampling_ratio = self.sampling_ratio)
            return user_cl_loss + item_cl_loss, negative_embedding_new
        
        else:
            return user_cl_loss + item_cl_loss


    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb= self.model.forward()

        if self.save_model:
            model_path = self.config.output_dir+self.config.model+ '@' + 'embedding.pth'
            torch.save(self.model.state_dict(), model_path)
            model_path = self.config.output_dir+self.config.model+ '@' + 'SGM.pth'
            torch.save(self.state_score['model'].state_dict(), model_path)

    def predict(self, u):
        u = self.data.get_user_id(u)
        score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
        return score.cpu().numpy()


class LightGCN(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LightGCN, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.emb_size))),
        })
        return embedding_dict

    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = []
        for k in range(self.n_layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.data.user_num, self.data.item_num])
        return user_all_embeddings, item_all_embeddings
       
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
    
class SGM(nn.Module):
  def __init__(self, data, emb_size, score_dim, config, act):
    super().__init__()

    self.embed_dim = emb_size
    encoder_dim = score_dim

    tdim = self.embed_dim*4
    self.act = get_act(act)

    modules = []

    modules.append(nn.Linear(emb_size, tdim))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    modules.append(nn.Linear(tdim, tdim))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)

    self.all_modules = nn.ModuleList(modules)

    dim_in = emb_size
    dim_out = encoder_dim[0]
    self.inputs = nn.Linear(dim_in, dim_out) # input layer
    self.encoder = layers.Encoder(encoder_dim, tdim, act) # encoder

    dim_in = encoder_dim[-1]
    dim_out = encoder_dim[-1]
    self.bottom_block = nn.Linear(dim_in, dim_out) #bottom_layer
    
    self.decoder = layers.Decoder(list(reversed(encoder_dim)), tdim, act) #decoder

    dim_in = encoder_dim[0]
    dim_out = emb_size
    self.outputs = nn.Linear(dim_in, dim_out) #output layer


  def forward(self, x, time_cond):

    modules = self.all_modules 
    m_idx = 0


    temb = layers.get_timestep_embedding(time_cond, self.embed_dim)
    temb = modules[m_idx](temb)
    m_idx += 1
    temb= self.act(temb)
    temb = modules[m_idx](temb)
    m_idx += 1
    
    inputs = self.inputs(x) #input layer
    skip_connections, encoding = self.encoder(inputs, temb)
    encoding = self.bottom_block(encoding)
    x = self.decoder(skip_connections, encoding, temb) 
    outputs = self.outputs(x)

    return outputs
  