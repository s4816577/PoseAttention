import math
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
###############################################################################
# Functions
###############################################################################


def weight_init_googlenet(key, module, weights=None):

    if key == "LSTM":
        for name, param in module.named_parameters():
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.xavier_normal_(param)
    elif weights is None:
        init.constant_(module.bias.data, 0.0)
        if key == "XYZ":
            init.normal_(module.weight.data, 0.0, 0.5)
        elif key == "LSTM":
            init.xavier_normal_(module.weight.data)
        else:
            init.normal_(module.weight.data, 0.0, 0.01)
    else:
        # print(key, weights[(key+"_1").encode()].shape, module.bias.size())
        module.bias.data[...] = torch.from_numpy(weights[(key+"_1").encode()])
        module.weight.data[...] = torch.from_numpy(weights[(key+"_0").encode()])
    return module

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_network(input_nc, lstm_hidden_size, model, init_from=None, isTest=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())

    if model == 'posenet':
        netG = PoseNet(input_nc, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    elif model == 'poselstm':
        netG = PoseLSTM(input_nc, lstm_hidden_size, weights=init_from, isTest=isTest, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Model name [%s] is not recognized' % model)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG

##############################################################################
# Classes
##############################################################################

# defines the regression heads for googlenet
class RegressionHead(nn.Module):
    def __init__(self, lossID, weights=None, lstm_hidden_size=None):
        super(RegressionHead, self).__init__()
        self.has_lstm = lstm_hidden_size != None
        self.has_attn = False
        self.has_mul_attn = True
        dropout_rate = 0.5 if lossID == "loss3" else 0.7
        nc_loss = {"loss1": 512, "loss2": 528}
        nc_cls = [1024, 2048] if lstm_hidden_size is None else [lstm_hidden_size*4, lstm_hidden_size*4]
        
        #for attn2
        self.attn_1 = nn.Linear(512, 512)
        self.attn_2 = nn.Linear(512, 1)
        self.attn_dropout = nn.Dropout(0.5)
        # inititalize
        nn.init.xavier_uniform_(self.attn_1.weight)
        nn.init.xavier_uniform_(self.attn_2.weight)
        self.attn_1.bias.data.fill_(0.0)
        self.attn_2.bias.data.fill_(0.0)
        
        if self.has_mul_attn:
            #for attn3
            self.attn_3 = nn.Linear(64, 64)
            self.attn_4 = nn.Linear(64, 1)
            #for multi_head_attn
            self.multi_head_linear = nn.Linear(512, 512)
            self.multi_final_linear = nn.Linear(512, 512)
            self.multi_dropout = nn.Dropout(0.5)
            self.multi_layer_norm = nn.LayerNorm(512)
            #init multi
            self.attn_3.bias.data.fill_(0.0)
            self.attn_4.bias.data.fill_(0.0)
            self.multi_head_linear.bias.data.fill_(0.0)
            self.multi_final_linear.bias.data.fill_(0.0)
            nn.init.xavier_uniform_(self.attn_3.weight)
            nn.init.xavier_uniform_(self.attn_4.weight)
            nn.init.xavier_uniform_(self.multi_head_linear.weight)
            nn.init.xavier_uniform_(self.multi_final_linear.weight)

        self.dropout = nn.Dropout(p=dropout_rate)
        if lossID != "loss3":
            self.projection = nn.Sequential(*[nn.AvgPool2d(kernel_size=5, stride=3),
                                              weight_init_googlenet(lossID+"/conv", nn.Conv2d(nc_loss[lossID], 128, kernel_size=1), weights),
                                              nn.ReLU(inplace=True)])
            self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet(lossID+"/fc", nn.Linear(2048, 1024), weights),
                                               nn.ReLU(inplace=True)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[0], 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[0], 4))
            if lstm_hidden_size is not None:
                self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
                self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
        else:
            self.projection = nn.AvgPool2d(kernel_size=7, stride=1)
            self.cls_fc_pose = nn.Sequential(*[weight_init_googlenet("pose", nn.Linear(1024, 2048)),
                                               nn.ReLU(inplace=True)])
            self.cls_fc_xy = weight_init_googlenet("XYZ", nn.Linear(nc_cls[1], 3))
            self.cls_fc_wpqr = weight_init_googlenet("WPQR", nn.Linear(nc_cls[1], 4))

            if lstm_hidden_size is not None:
                self.lstm_pose_lr = weight_init_googlenet("LSTM", nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
                self.lstm_pose_ud = weight_init_googlenet("LSTM", nn.LSTM(input_size=32, hidden_size=lstm_hidden_size, bidirectional=True, batch_first=True))
    
    def attention(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [TxBxK]
        # Values = [TxBxV]
        # Outputs = a:[TxB], lin_comb:[BxV]

        # Here we assume q_dim == k_dim (dot product attention)
        scale = 1. / math.sqrt(query.size(1))
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(scale), dim=2) # scale, normalize

        values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return energy, linear_combination
        
    def attention2(self, x, return_attention=False):
        """
        Input x is encoder output
        return_attention decides whether to return
        attention scores over the encoder output
        """
        sequence_length = x.shape[1]
        self_attention_scores = self.attn_2(torch.tanh(self.attn_1(x)))
        
        # Attend for each time step using the previous context
        context_vectors = []
        attention_vectors = []

        for t in range(sequence_length):
            # For each timestep the context that is attented grows
            # as there are more available previous hidden states
            weighted_attention_scores = F.softmax(
                self_attention_scores[:, :t + 1, :].clone(), dim=1)

            context_vectors.append(
                torch.sum(weighted_attention_scores * x[:, :t + 1, :].clone(), dim=1))

            if return_attention:
                attention_vectors.append(
                    weighted_attention_scores.cpu().detach().numpy())

        context_vectors = torch.stack(context_vectors).transpose(0, 1)

        return context_vectors, attention_vectors
        
    def attention3(self, x, return_attention=False):
        """
        Input x is encoder output
        return_attention decides whether to return
        attention scores over the encoder output
        """
        sequence_length = x.shape[1]
        self_attention_scores = self.attn_4(torch.tanh(self.attn_3(x)))
        
        # Attend for each time step using the previous context
        context_vectors = []
        attention_vectors = []

        for t in range(sequence_length):
            # For each timestep the context that is attented grows
            # as there are more available previous hidden states
            weighted_attention_scores = F.softmax(
                self_attention_scores[:, :t + 1, :].clone(), dim=1)

            context_vectors.append(
                torch.sum(weighted_attention_scores * x[:, :t + 1, :].clone(), dim=1))

            if return_attention:
                attention_vectors.append(
                    weighted_attention_scores.cpu().detach().numpy())

        context_vectors = torch.stack(context_vectors).transpose(0, 1)

        return context_vectors, attention_vectors
    
    def multi_head_attn(self, x):
        """ based on attention2 """
        num_heads = 8
        dim_per_head = int(512 / num_heads)
        batch_size = x.size(0)
        residual = x
        
        lineared_vectors = self.multi_head_linear(x)
        lineared_vectors = lineared_vectors.contiguous().view(batch_size * num_heads, -1, dim_per_head)
        context_vectors, _ = self.attention3(lineared_vectors)
        context_vectors = context_vectors.contiguous().view(batch_size, -1, dim_per_head * num_heads)
        
        context_vectors = self.multi_final_linear(context_vectors)
        context_vectors = self.multi_dropout(context_vectors)
        context_vectors = self.multi_layer_norm(residual + context_vectors)
        
        return context_vectors
    
    def forward(self, input):
        output = self.projection(input)
        output = self.cls_fc_pose(output.view(output.size(0), -1))
        if self.has_lstm:
            output = output.view(output.size(0), 32, -1)
            output_lr, (hidden_state_lr, cell_lr) = self.lstm_pose_lr(output.permute(0,1,2))
            output_ud, (hidden_state_ud, cell_ud) = self.lstm_pose_ud(output.permute(0,2,1))
            if self.has_attn:
                '''
                _, context_vector_lr = self.attention(torch.cat([cell_lr[-1], cell_lr[-2]], dim=1), output_lr.transpose(0,1), output_lr.transpose(0,1))
                _, context_vector_ud = self.attention(torch.cat([cell_ud[-1], cell_ud[-2]], dim=1), output_ud.transpose(0,1), output_ud.transpose(0,1))
                output = torch.cat([context_vector_lr, context_vector_ud], dim=1)
                '''
                
                context_vector_lr, _ = self.attention2(output_lr)
                context_vector_ud, _ = self.attention2(output_ud)
                output = torch.cat([context_vector_lr[:,-1,:], context_vector_ud[:,-1,:]], dim=1)
                
            elif self.has_mul_attn:
                context_vector_lr = self.multi_head_attn(output_lr)
                context_vector_ud = self.multi_head_attn(output_ud)
                output = torch.cat([context_vector_lr[:,-1,:], context_vector_ud[:,-1,:]], dim=1)
            else:
                output = torch.cat((hidden_state_lr[0,:,:],
                                    hidden_state_lr[1,:,:],
                                    hidden_state_ud[0,:,:],
                                    hidden_state_ud[1,:,:]), 1)
        output = self.dropout(output)
        output_xy = self.cls_fc_xy(output)
        output_wpqr = self.cls_fc_wpqr(output)
        output_wpqr = F.normalize(output_wpqr, p=2, dim=1)
        return [output_xy, output_wpqr]

# define inception block for GoogleNet
class InceptionBlock(nn.Module):
    def __init__(self, incp, input_nc, x1_nc, x3_reduce_nc, x3_nc, x5_reduce_nc,
                 x5_nc, proj_nc, weights=None, gpu_ids=[]):
        super(InceptionBlock, self).__init__()
        self.gpu_ids = gpu_ids
        # first
        self.branch_x1 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/1x1", nn.Conv2d(input_nc, x1_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x3 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/3x3_reduce", nn.Conv2d(input_nc, x3_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/3x3", nn.Conv2d(x3_reduce_nc, x3_nc, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True)])

        self.branch_x5 = nn.Sequential(*[
            weight_init_googlenet("inception_"+incp+"/5x5_reduce", nn.Conv2d(input_nc, x5_reduce_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("inception_"+incp+"/5x5", nn.Conv2d(x5_reduce_nc, x5_nc, kernel_size=5, padding=2), weights),
            nn.ReLU(inplace=True)])

        self.branch_proj = nn.Sequential(*[
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            weight_init_googlenet("inception_"+incp+"/pool_proj", nn.Conv2d(input_nc, proj_nc, kernel_size=1), weights),
            nn.ReLU(inplace=True)])

        if incp in ["3b", "4e"]:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.pool = None

    def forward(self, input):
        outputs = [self.branch_x1(input), self.branch_x3(input),
                   self.branch_x5(input), self.branch_proj(input)]
        # print([[o.size()] for o in outputs])
        output = torch.cat(outputs, 1)
        if self.pool is not None:
            return self.pool(output)
        return output

class PoseNet(nn.Module):
    def __init__(self, input_nc, weights=None, isTest=False,  gpu_ids=[]):
        super(PoseNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.isTest = isTest
        self.before_inception = nn.Sequential(*[
            weight_init_googlenet("conv1/7x7_s2", nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            weight_init_googlenet("conv2/3x3_reduce", nn.Conv2d(64, 64, kernel_size=1), weights),
            nn.ReLU(inplace=True),
            weight_init_googlenet("conv2/3x3", nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            ])

        self.inception_3a = InceptionBlock("3a", 192, 64, 96, 128, 16, 32, 32, weights, gpu_ids)
        self.inception_3b = InceptionBlock("3b", 256, 128, 128, 192, 32, 96, 64, weights, gpu_ids)
        self.inception_4a = InceptionBlock("4a", 480, 192, 96, 208, 16, 48, 64, weights, gpu_ids)
        self.inception_4b = InceptionBlock("4b", 512, 160, 112, 224, 24, 64, 64, weights, gpu_ids)
        self.inception_4c = InceptionBlock("4c", 512, 128, 128, 256, 24, 64, 64, weights, gpu_ids)
        self.inception_4d = InceptionBlock("4d", 512, 112, 144, 288, 32, 64, 64, weights, gpu_ids)
        self.inception_4e = InceptionBlock("4e", 528, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5a = InceptionBlock("5a", 832, 256, 160, 320, 32, 128, 128, weights, gpu_ids)
        self.inception_5b = InceptionBlock("5b", 832, 384, 192, 384, 48, 128, 128, weights, gpu_ids)

        self.cls1_fc = RegressionHead(lossID="loss1", weights=weights)
        self.cls2_fc = RegressionHead(lossID="loss2", weights=weights)
        self.cls3_fc = RegressionHead(lossID="loss3", weights=weights)

        self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                   self.inception_4a, self.inception_4b,
                                   self.inception_4c, self.inception_4d,
                                   self.inception_4e, self.inception_5a,
                                   self.inception_5b, self.cls1_fc,
                                   self.cls2_fc, self.cls3_fc
                                   ])
        if self.isTest:
            self.model.eval() # ensure Dropout is deactivated during test

    def forward(self, input):

        output_bf = self.before_inception(input)
        output_3a = self.inception_3a(output_bf)
        output_3b = self.inception_3b(output_3a)
        output_4a = self.inception_4a(output_3b)
        output_4b = self.inception_4b(output_4a)
        output_4c = self.inception_4c(output_4b)
        output_4d = self.inception_4d(output_4c)
        output_4e = self.inception_4e(output_4d)
        output_5a = self.inception_5a(output_4e)
        output_5b = self.inception_5b(output_5a)

        if not self.isTest:
            return self.cls1_fc(output_4a) + self.cls2_fc(output_4d) +  self.cls3_fc(output_5b)
        return self.cls3_fc(output_5b)

class PoseLSTM(PoseNet):
    def __init__(self, input_nc, lstm_hidden_size, weights=None, isTest=False,  gpu_ids=[]):
            super(PoseLSTM, self).__init__(input_nc, weights, isTest, gpu_ids)
            self.cls1_fc = RegressionHead(lossID="loss1", weights=weights, lstm_hidden_size=lstm_hidden_size)
            self.cls2_fc = RegressionHead(lossID="loss2", weights=weights, lstm_hidden_size=lstm_hidden_size)
            self.cls3_fc = RegressionHead(lossID="loss3", weights=weights, lstm_hidden_size=lstm_hidden_size)

            self.model = nn.Sequential(*[self.inception_3a, self.inception_3b,
                                       self.inception_4a, self.inception_4b,
                                       self.inception_4c, self.inception_4d,
                                       self.inception_4e, self.inception_5a,
                                       self.inception_5b, self.cls1_fc,
                                       self.cls2_fc, self.cls3_fc
                                       ])
            if self.isTest:
                self.model.eval() # ensure Dropout is deactivated during test
