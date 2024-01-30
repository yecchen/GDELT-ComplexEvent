import torch
import torch.nn as nn

from models.decoder import ConvTransE
from models.kernel import RGCN

class LoGo_share(nn.Module):
    def __init__(self, conf):
        super(LoGo_share, self).__init__()

        self.conf = conf
        num_ents = conf["num_ents"]
        num_rels = conf["num_rels"]
        h_dim = conf["h_dim"]

        self.ent_embs = torch.nn.Parameter(torch.FloatTensor(num_ents, h_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.ent_embs)

        self.rel_embs = torch.nn.Parameter(torch.FloatTensor(num_rels * 2, h_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.rel_embs)

        self.rgcn = RGCN(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             n_bases=conf["n_bases"],
                             n_layers=conf["n_layers"],
                             dropout=conf["dropout"],
                             self_loop=conf["self_loop"],
                             skip_connect=conf["skip_connect"])

        self.hist_gru = nn.GRU(h_dim, h_dim, batch_first=True) 

        # decoder
        self.decoder = ConvTransE(num_ents, h_dim, conf["input_dropout"], conf["hidden_dropout"], conf["feat_dropout"])

        self.loss_e = torch.nn.CrossEntropyLoss()


    def predict_global(self, g_list, triplets):
        history_embs = []
        for i, g in enumerate(g_list):
            h = self.rgcn.forward(g, self.ent_embs, self.rel_embs)
            history_embs.append(h)

        history_embs = torch.stack(history_embs, dim=1) # [num_ents, hist_len, h_dim]
        _, his_rep = self.hist_gru(history_embs)
        ent_rep = his_rep.squeeze(0)

        scores, query_emb = self.decoder.forward(ent_rep, self.rel_embs, triplets)
        return scores, query_emb, ent_rep

    def predict_local(self, g_list, triplets):
        history_embs = []
        for i, g in enumerate(g_list):
            h = self.rgcn.forward(g, self.ent_embs, self.rel_embs)
            history_embs.append(h)

        history_embs = torch.stack(history_embs, dim=1) # [num_ents, hist_len, h_dim]
        _, his_rep = self.hist_gru(history_embs)
        ent_rep = his_rep.squeeze(0)

        scores, query_emb = self.decoder.forward(ent_rep, self.rel_embs, triplets)
        return scores, query_emb, ent_rep

    def forward_query(self, query_embs, ent_embs, triplets):
        scores = self.predict_query(query_embs, ent_embs)
        loss = self.loss_e(scores, triplets[:, 2])
        return loss

    def predict_query(self, query_embs, ent_embs):
        scores = torch.mm(query_embs, ent_embs.transpose(1, 0))
        scores = scores.view(-1, self.conf["num_ents"])
        return scores

    def get_global_ent_embs(self, g_list):
        history_embs = []
        for i, g in enumerate(g_list):
            h = self.rgcn.forward(g, self.ent_embs, self.rel_embs)
            history_embs.append(h)

        history_embs = torch.stack(history_embs, dim=1) # [num_ents, hist_len, h_dim]
        _, his_rep = self.hist_gru(history_embs)
        ent_rep = his_rep.squeeze(0)
        return ent_rep

    def get_local_ent_embs(self, g_list):
        history_embs = []
        for i, g in enumerate(g_list):
            h = self.rgcn.forward(g, self.ent_embs, self.rel_embs)
            history_embs.append(h)

        history_embs = torch.stack(history_embs, dim=1) # [num_ents, hist_len, h_dim]
        _, his_rep = self.hist_gru(history_embs)
        ent_rep = his_rep.squeeze(0)
        return ent_rep

    def get_query_embs(self, s_embs, triplets):
        query_embs = self.decoder.forward_s(s_embs, self.rel_embs, triplets)
        return query_embs
