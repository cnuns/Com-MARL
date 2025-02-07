import torch
import torch.nn as nn

class AttentionModule(nn.Module):
    """ Applies attention mechanism on the `context` using the `query`.

    Args:
        dimensions (int): Dimensionality of the query and context.
        attention_type (str, optional): How to compute the attention score:

            * dot: :math:`score(H_j,q) = H_j^T q`
            * general: :math:`score(H_j, q) = H_j^T W_a q`
    """

    def __init__(self, dimensions, attention_type='general'):
        super().__init__()

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        elif self.attention_type == 'diff':
            self.linear_in = nn.Linear(dimensions, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, loss=None):
        """
            Self attention

            n_paths, max_path_length, n_agents, emb_feat_dim = query.shape
            OR
            bs, n_agents, emb_feat_dim = query.shape
            OR
            n_agents, emb_feat_dim = query.shape

        """

        if self.attention_type in ['general', 'dot']:
            context = query.transpose(-2, -1).contiguous()
            if self.attention_type == 'general':
                query = self.linear_in(query)
            attention_scores = torch.matmul(query, context)

            if loss is not None:
                # (1,n,n) -> (1,l,n,n)
                attention_scores = attention_scores.expand_as(loss)
                attention_scores.masked_fill_(loss, -1e10)

            attention_weights = self.softmax(attention_scores)

        return attention_weights


