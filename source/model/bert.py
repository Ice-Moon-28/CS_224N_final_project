import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from source.model.base_bert import BertPreTrainedModel
from source.util.utils import *

class BertSelfAttention(nn.Module):
  debug = False

  def __init__(self, config):
    super().__init__()

    # self.debug = False

    self.attention_head_num = config.num_attention_heads

    self.attention_hidden_size = int(config.hidden_size / config.num_attention_heads)

    self.all_head_size = self.attention_head_num * self.attention_hidden_size

    self.query = nn.Linear(config.hidden_size, self.all_head_size)

    self.key = nn.Linear(config.hidden_size, self.all_head_size)

    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    self.dropout = nn.Dropout(config.attention_dropout_capacity)

    self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    self.add_norm = AddAndNormLayer(config)

  def transform_into_multi_heads(self, state: Tensor, linear_layers: nn.Linear):
    """
       input: (batch_size, seq_len, hidden_size)
       linear_layers: (hidden_size, all_head_size)

       output: (batch_size, num_attention_heads, seq_len, attention_head_size)
    """
    batch_size, seq_len = state.shape[:2]

    state = linear_layers(state)

    # there is significant diff between view function and transpose function

    transformed_state = state.view(batch_size, seq_len, self.attention_head_num, self.attention_hidden_size)

    return transformed_state.transpose(1, 2)
   

  def attention(self, hidden_states, attention_mask=None):
    query = self.transform_into_multi_heads(hidden_states, self.query)

    key = self.transform_into_multi_heads(hidden_states, self.key)

    value = self.transform_into_multi_heads(hidden_states, self.value)

    attention_value = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.attention_hidden_size)

    batch_size, _, seq_len  = attention_value.shape[:3]

    if attention_mask is not None:
      attention_value = attention_value + attention_mask.repeat(1, 1, seq_len, 1)

    attention_value = nn.Softmax(dim=-1)(attention_value)
    # import pdb
    # pdb.set_trace()

    attention_hidden_state = torch.matmul(attention_value, value)

    attention_hidden_state = attention_hidden_state.transpose(1, 2).contiguous().view(batch_size, seq_len, self.all_head_size)

    return attention_hidden_state
  
  def forward(self, hidden_states, attention_mask):

    attention_hidden_states = self.attention(hidden_states, attention_mask)

    attention_hidden_states = self.dense(attention_hidden_states)

    attention_hidden_states = self.dropout(attention_hidden_states)

    attention_hidden_states = self.add_norm(hidden_states, attention_hidden_states)

    return attention_hidden_states
  

class AddAndNormLayer(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()

    self.normalized_layer = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)



  def forward(self, input, output):

    return self.normalized_layer(input + output)



class FeedForwardLayer(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()

    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)

    self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    self.active_fn = F.gelu

    self.add_norm = AddAndNormLayer(config=config)

  def forward(self, hidden_states):

    intermediate_output = self.interm_dense(hidden_states)

    intermediate_output = self.active_fn(intermediate_output)

    output = self.output_dense(intermediate_output)

    output = self.output_dropout(output)

    output = self.add_norm(hidden_states, output)

    return output
  

class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.self_attention = BertSelfAttention(config)

    self.feed_forward = FeedForwardLayer(config)

  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first BERT layer) or from the previous BERT layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf.
    Each block consists of:
    1. A multi-head attention layer (BertSelfAttention).
    2. An add-norm operation that takes the input and output of the multi-head attention layer.
    3. A feed forward layer.
    4. An add-norm operation that takes the input and output of the feed forward layer.
    """
    # 1. Multi-head attention layer
    attention_output = self.self_attention(hidden_states, attention_mask)

    output = self.feed_forward(attention_output)

    return output



class BertModel(BertPreTrainedModel):
  """
  The BERT model returns the final embeddings for each token in a sentence.
  
  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n BERT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # BERT encoder.
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get word embedding from self.word_embedding into input_embeds.
    inputs_embeds = self.word_embedding(input_ids)

    # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)

    # Get token type ids. Since we are not considering token type, this embedding is
    # just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    embeds = inputs_embeds + pos_embeds + tk_type_embeds
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)

    return embeds
    # # Get word embedding from self.word_embedding into input_embeds.
    # inputs_embeds = None
    # ### TODO
    # raise NotImplementedError


    # # Use pos_ids to get position embedding from self.pos_embedding into pos_embeds.
    # pos_ids = self.position_ids[:, :seq_length]
    # pos_embeds = None
    # ### TODO
    # raise NotImplementedError


    # # Get token type ids. Since we are not considering token type, this embedding is
    # # just a placeholder.
    # tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    # tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    # ### TODO
    # raise NotImplementedError


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.bert_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """

    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of BertLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    """
        pooled output 是一个固定长度的向量，通常是 768 维的向量。这个向量代表了整个输入序列的语义信息，可以作为下游任务的输入。

        总的来说, BERT 在这里做这样的操作是为了得到一个固定长度的向量表示，来代表整个输入序列，并且降低计算复杂度和存储空间的需求。
    """

    # Get cls token hidden state.
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}

