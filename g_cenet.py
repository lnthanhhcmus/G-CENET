import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import copy
import numpy as np

class GENERATE_MODE(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(GENERATE_MODE, self).__init__()

        self.W_mlp = nn.Linear(embedding_dim * 3, output_dim)

        print("GENERATE MODE Initiated")

    def forward(self, ent_embed, rel_embed, tim_embed, entity):
        if entity == 'object':
            m_t = torch.cat((ent_embed, rel_embed, tim_embed), dim=1)
        if entity == 'subject':
            m_t = torch.cat((rel_embed, ent_embed, tim_embed), dim=1)

        generate_score = self.W_mlp(m_t)

        return F.softmax(generate_score, dim=1)


class CENET(nn.Module):
    def __init__(self, num_e, num_rel, num_t, args):
        super(CENET, self).__init__()
        # stats
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.args = args

        self.contrastive_hidden_layer = nn.Linear(3 * args.embedding_dim, args.embedding_dim)
        self.contrastive_output_layer = nn.Linear(args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.weights_init(self.linear_pred_layer_1)
        self.weights_init(self.linear_pred_layer_2)

        self.dropout = nn.Dropout(args.dropout)
        self.logSoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.crossEntropy = nn.BCELoss()
        # self.oracle_mode = args.oracle_mode

        print('CENET Initiated')

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, batch_block, mode_lk, entity_embeds, 
                rel_embeds, history_tag, non_history_tag, 
                entity, frequency_hidden, total_data=None):
    
        quadruples, s_history_event_o, o_history_event_s, \
        s_history_label_true, o_history_label_true, _, _ = batch_block

        if isListEmpty(s_history_event_o) or isListEmpty(o_history_event_s):
            sub_rank, obj_rank, batch_loss = [None] * 3
            if mode_lk == 'Training':
                return batch_loss
            elif mode_lk in ['Valid', 'Test']:
                return sub_rank, batch_loss
            else:
                return None

        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]

        if mode_lk == 'Training':

            if entity == 'object':
                nce_loss, nce_score = self.calculate_nce_loss(o, s, r, entity_embeds, rel_embeds, 
                                                      self.linear_pred_layer_1, 
                                                      self.linear_pred_layer_2,
                                                      history_tag,
                                                      non_history_tag
                                                      )
                
                spc_loss = self.calculate_spc_loss(o, r, entity_embeds, rel_embeds,
                                                      s_history_label_true,
                                                      frequency_hidden
                                                      )

            if entity == 'subject':
                nce_loss, nce_score = self.calculate_nce_loss(s, o, r, entity_embeds, rel_embeds,
                                                      self.linear_pred_layer_1,
                                                      self.linear_pred_layer_2,
                                                      history_tag,
                                                      non_history_tag
                                                      )
                
                spc_loss = self.calculate_spc_loss(s, r, entity_embeds, rel_embeds,
                                                      o_history_label_true,
                                                      frequency_hidden
                                                      )
            
            return self.args.alpha * nce_loss + (1 - self.args.alpha) * spc_loss, nce_score
        
        elif mode_lk in ['Valid', 'Test']:

            if entity == 'object':

                nce_loss, nce_score = self.calculate_nce_loss(o, s, r, entity_embeds, rel_embeds,
                                                              self.linear_pred_layer_1,
                                                              self.linear_pred_layer_2,
                                                              history_tag,
                                                              non_history_tag
                                                              )

            if entity == 'subject':

                nce_loss, nce_score = self.calculate_nce_loss(s, o, r, entity_embeds, rel_embeds,
                                                              self.linear_pred_layer_1,
                                                              self.linear_pred_layer_2,
                                                              history_tag,
                                                              non_history_tag
                                                              )

            return nce_loss, nce_score

    def calculate_nce_loss(self, actor1, actor2, r, entity_embeds, rel_embeds, linear1, linear2, history_tag, non_history_tag):
        # preds_raw1 = self.tanh(linear1(
        #     self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))
        preds_raw1 = self.tanh(linear1(
            self.dropout(torch.cat((entity_embeds[actor1], rel_embeds[r]), dim=1))))
        # preds1 = F.softmax(preds_raw1.mm(self.entity_embeds.transpose(0, 1)) + history_tag, dim=1)
        preds1 = F.softmax(preds_raw1.mm(entity_embeds.transpose(0, 1)) + history_tag, dim=1)

        # preds_raw2 = self.tanh(linear2(
        #     self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))))
        preds_raw2 = self.tanh(linear2(
            self.dropout(torch.cat((entity_embeds[actor1], rel_embeds[r]), dim=1))))
        # preds2 = F.softmax(preds_raw2.mm(self.entity_embeds.transpose(0, 1)) + non_history_tag, dim=1)
        preds2 = F.softmax(preds_raw2.mm(entity_embeds.transpose(0, 1)) + non_history_tag, dim=1)

        # cro_entr_loss = self.criterion_link(preds1 + preds2, actor2)

        nce = torch.sum(torch.gather(torch.log(preds1 + preds2), 1, actor2.view(-1, 1).to(torch.int64)))
        nce /= -1. * actor2.shape[0]

        # pred_actor2 = torch.argmax(preds1 + preds2, dim=1)  # predicted result
        # correct = torch.sum(torch.eq(pred_actor2, actor2))
        # if correct.shape[0] == None:
        #     correct = [0]
        # accuracy = 1. * correct.item() / actor2.shape[0]
        # print('# Batch accuracy', accuracy)

        return nce, preds1 + preds2

    # contrastive
    # def freeze_parameter(self):
    #     self.linear_pred_layer_1.requires_grad_(False)
    #     self.linear_pred_layer_2.requires_grad_(False)
    #     self.linear_frequency.requires_grad_(False)
    #     self.contrastive_hidden_layer.requires_grad_(False)
    #     self.contrastive_output_layer.requires_grad_(False)

    def contrastive_layer(self, x):
        # Implement from the encoder E to the projection network P
        # x = F.normalize(x, dim=1)
        x = self.contrastive_hidden_layer(x)
        # x = F.relu(x)
        # x = self.contrastive_output_layer(x)
        # Normalize to unit hypersphere
        # x = F.normalize(x, dim=1)
        return x

    def calculate_spc_loss(self, actor1, r, entity_embeds, rel_embeds, targets, frequency_hidden):
        # projections = self.contrastive_layer(
        #     torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1))
        
        projections = self.contrastive_layer(
            torch.cat((entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1))
        targets = torch.squeeze(targets)
        """if np.random.randint(0, 10) < 1 and torch.sum(targets) / targets.shape[0] < 0.65 and torch.sum(targets) / targets.shape[0] > 0.35:
            np.savetxt("xx.tsv", projections.detach().cpu().numpy(), delimiter="\t")
            np.savetxt("yy.tsv", targets.detach().cpu().numpy(), delimiter="\t")
        """
        dot_product_tempered = torch.mm(projections, projections.T) / 1.0
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_similar_class = to_device(targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets)
        mask_anchor_out = to_device(1 - torch.eye(exp_dot_tempered.shape[0]))
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        # prob = exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True))
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        # log_prob = -torch.log(prob)
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        if torch.any(torch.isnan(supervised_contrastive_loss)):
            return 0
        return supervised_contrastive_loss


class G_CENET(nn.Module):

    def __init__(self, num_e, num_rel, num_t, args):
        super(G_CENET, self).__init__()
        
        #The code will be available after passing the first round of review.
        

        print("CENET GENERATE Initiated")

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
    
    def forward(self, batch_block, mode_lk, entity, total_data=None):
        #The code will be available after passing the first round of review.
            

    def regularization_loss(self, reg_param):
        #The code will be available after passing the first round of review.        
    
    def link_predict(self, nce_loss, preds, ce_loss, actor1, actor2, r, all_triples, pred_known):
        #The code will be available after passing the first round of review.

    def get_init_time(self, quadrupleList):
            #The code will be available after passing the first round of review.
    
    def get_raw_m_t(self, quadrupleList):
        #The code will be available after passing the first round of review.
    
    def get_raw_m_t_sub(self, quadrupleList):
        #The code will be available after passing the first round of review.