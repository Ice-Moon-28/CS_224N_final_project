import torch
from bert import BertModel


sanity_data = torch.load("./sanity_check.data")
sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                         [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],[1, 1, 1, 1, 1, 1, 1, 1]])

# Load model.
bert = BertModel.from_pretrained('bert-base-uncased')

outputs = bert(sent_ids, att_mask)
att_mask = att_mask.unsqueeze(-1)

print(att_mask.shape, outputs['last_hidden_state'].shape)

outputs['last_hidden_state'] = outputs['last_hidden_state'] * att_mask
sanity_data['last_hidden_state'] = sanity_data['last_hidden_state'] * att_mask


for k in ['last_hidden_state', 'pooler_output']:
    print(k, '====== key ======') 
    """
        tensor1 和 tensor2 是两个需要比较的张量。
        atol 是绝对容忍度 (absolute tolerance), 默认值为 1e-8。
        rtol 是相对容忍度 (relative tolerance), 默认值为 1e-5。
        函数返回一个布尔值，表示两个张量是否在指定的容忍度内相等。

        在你的例子中, torch.allclose 函数用于检查 outputs[k] 和 sanity_data[k] 两个张量是否在绝对容忍度 1e-5 和相对容忍度 1e-3 内相等。
    """

    print(outputs[k], sanity_data[k])
    assert torch.allclose(outputs[k], sanity_data[k], atol=1e-5, rtol=1e-3)
print("Your BERT implementation is correct!")
