
import math
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True, fusion=False):

        super(ConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.w = torch.nn.Parameter(torch.Tensor(embedding_dim * 2, embedding_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w)

        if fusion:
            self.conv1 = torch.nn.Conv1d(4, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        else:
            self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))  # kernel size is odd, then padding = math.floor(kernel_size/2)
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn0_fusion = torch.nn.BatchNorm1d(4)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)
        self.bn3 = torch.nn.BatchNorm1d(embedding_dim)
        # self.bn4 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn_init = torch.nn.BatchNorm1d(embedding_dim)


    def forward(self, evolve_ent, evolve_rel, triplets):
        #embedded_all_evovle = F.tanh(evolve_ent)
        embedded_all_evovle = evolve_ent
        batch_size = len(triplets)
        e1_embedded_evolve = embedded_all_evovle[triplets[:, 0]].unsqueeze(1)
        rel_embedded_evolve = evolve_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded_evolve, rel_embedded_evolve], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        query = F.relu(x)
        x = torch.mm(query, embedded_all_evovle.transpose(1, 0))

        return x, query

    def forward_s(self, e1_embs, evolve_rel, triplets):
        batch_size = len(triplets)
        e1_embedded_evolve = e1_embs.unsqueeze(1)
        rel_embedded_evolve = evolve_rel[triplets[:, 1]].unsqueeze(1)
        stacked_inputs = torch.cat([e1_embedded_evolve, rel_embedded_evolve], 1)
        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if batch_size > 1:
            x = self.bn2(x)
        query = F.relu(x)
        return query