
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, comp='sub'):
        super(RGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.skip_connect = skip_connect
        self.comp = comp

        # WL
        self.weight_neighbor = self.get_param([in_feat, out_feat])

        if self.self_loop:
            self.loop_weight = self.get_param([in_feat, out_feat])

        if self.skip_connect:
            self.skip_connect_weight =self.get_param([in_feat, out_feat])
            self.skip_connect_bias = self.get_param([out_feat])
            nn.init.zeros_(self.skip_connect_bias)

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
        return param


    def forward(self, g, prev_h, emb_rel):
        # should excecute before the propagate
        if self.self_loop:
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)

        node_repr = self.propagate(g, emb_rel)

        if self.self_loop:
            node_repr = node_repr + loop_message
        if prev_h is not None and self.skip_connect:
            skip_weight = torch.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr

        return node_repr


    def propagate(self, g, emb_rel):
        g.update_all(lambda x: self.msg_func(x, emb_rel), fn.sum(msg='msg', out='h'), self.apply_func)
        return g.ndata['h']


    def msg_func(self, edges, emb_rel):
        relation = emb_rel.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)
        if self.comp == "sub":
            msg = node + relation
        elif self.comp == "mult":
            msg = node * relation
        msg = torch.mm(msg, self.weight_neighbor)

        return {'msg': msg}


    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, n_bases=-1, n_layers=1, dropout=0, act=F.rrelu, self_loop=False, skip_connect=False, comp="sub"):
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.n_bases = n_bases
        self.n_layers = n_layers
        self.dropout = dropout
        self.skip_connect = skip_connect
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.act = act
        self.comp = comp
        self.build_model()
        print('use kernel: RGCN')


    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.n_layers):
            if self.skip_connect:
                sc = False if idx == 0 else True
            else:
                sc = False
            h2h = RGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.n_bases, activation=self.act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, comp=self.comp)
            self.layers.append(h2h)


    def forward(self, g, init_ent_emb, init_rel_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        x, r = init_ent_emb, init_rel_emb
        prev_h = None
        for i, layer in enumerate(self.layers):
            prev_h = layer(g, prev_h, r)

        return prev_h