import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.utils import rnn
from transformers import AutoTokenizer
import clip
from torch.cuda.amp import autocast

class MultiHead(nn.Module):

    def __init__(self, emd_dim, d_model, head):
        super(MultiHead, self).__init__()

        self.d_model = d_model
        self.head = head
        self.qmat = nn.Linear(emd_dim, d_model)
        self.kmat = nn.Linear(emd_dim, d_model)
        self.vmat = nn.Linear(emd_dim, d_model)
        self.omat = nn.Linear(d_model, emd_dim)

    def make_heads(self, x):
        return x.view(x.size()[0], x.size()[1], self.head, self.d_model // self.head).transpose(1, 2)

    def forward(self, x):

        q, k, v = self.qmat(x), self.kmat(x), self.vmat(x)
        q, k, v = self.make_heads(q), self.make_heads(k), self.make_heads(v)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(x.size()[0], -1, self.d_model)

        return self.omat(x)

class MLP(nn.Module):
    
    def __init__(self, emd_dim):
        super(MLP, self).__init__()
        self.emd_dim = emd_dim
        self.ff = nn.Sequential(
            nn.Linear(self.emd_dim, self.emd_dim * 4),
            nn.GELU(),
            nn.Linear(self.emd_dim * 4, self.emd_dim)
        )
        
    def forward(self, input_tensor): 
        return self.ff(input_tensor)

class BLOCK(nn.Module):

    def __init__(self, emd_dim, d_model, heads):
        super(BLOCK, self).__init__()

        self.norm1 = nn.LayerNorm(emd_dim)
        self.multihead = MultiHead(emd_dim, d_model, heads)
        self.norm2 = nn.LayerNorm(emd_dim)
        self.ff = MLP(emd_dim)

    def forward(self, x):
        
        x = x + self.multihead(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        
        return x

class ResBlocks(nn.Module):
    
    def __init__(self, inchannels, outchannels, num=1):
        super(ResBlocks, self).__init__()
        
        if num != 0:
            inchannels = outchannels
        self.linear1 = nn.Linear(inchannels, outchannels)
        self.linear2 = nn.Linear(outchannels, outchannels)
        self.norm1 = nn.LayerNorm(inchannels)
        self.norm2 = nn.LayerNorm(outchannels)
        self.shortcut = nn.Linear(inchannels, outchannels)
        
    def forward(self, x):
        
        shortcut = self.shortcut(x)
        x = self.norm1(x)
        out = torch.relu(self.linear1(x))
        x = self.norm2(out)
        out = self.linear2(out)
        
        return torch.relu(out + shortcut)
    
class Projection(nn.Module):
    
    def __init__(self, 
                 clip,
                 clip_embed=512,
                 phi_embed=3072,
                 num_layers=2,
                 num_attention_layers=4
        ):
        super(Projection, self).__init__()
        self.layers = nn.ModuleList([ResBlocks(clip_embed, phi_embed, i) for i in range(num_layers)])
        self.attention_layers = nn.ModuleList([BLOCK(phi_embed, 2048, 8) for i in range(num_attention_layers)])
        self.clip = clip
        
    def forward(self, image):
        
        image = image.to('cuda')
        image = image.type(torch.float16)
        
        with torch.no_grad(), autocast():
            x = self.clip.encode_image(image)

        x = x.type(torch.float16)
        
        for layer in self.layers:
            x = layer(x)
        x = x.unsqueeze(1)
        for layer in self.attention_layers:
            x = layer(x)
        
        return x

class DaVa(nn.Module):
    
    def __init__(self,
        phi_model,
        clip_embed=512, 
        phi_embed=3072, 
        num_layers=6,
        stage=1
    ):
        super(DaVa, self).__init__()
        
        self.EOS_TOKEN_ID    = 50256
        self.IMAGE_TOKEN_ID  = 23893
        
        self.stage=stage
        self.clip_embed=clip_embed
        self.phi_embed=phi_embed
        self.num_layers=num_layers
        self.device = torch.device('cuda')        

        clipped, clippreprocess = clip.load("ViT-B/32", device='cuda')
        self.clip = clip
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.phi = phi_model
        self.proj = Projection(clipped, self.clip_embed, self.phi_embed, self.num_layers).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
        self.max_length = 512
        
    def freeze(self):
        for p in self.phi.parameters():
            p.requires_grad = False
            
    def unfreeze(self):
        for p in self.phi.parameters():
            p.requires_grad = True
            
    #https://github.com/NExT-GPT/NExT-GPT/blob/main/code/model/anyToImageVideoAudio.py#L272
    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids.to(self.device)  # bsz x s2
        target_ids = target_ids.to(self.device)  # bsz x s2
        attention_mask = attention_mask.to(self.device)  # bsz x s2

        batch_size = input_ids.shape[0]

        bos = torch.ones([batch_size, 1], dtype=input_ids.dtype,
                            device=input_ids.device) * self.tokenizer.bos_token_id  # bsz x 1

        p_before = '### Human: <Img>'
        p_before_tokens = self.tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(self.device)

        bos_embeds = self.phi.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        p_before_embeds = self.phi.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
        p_after_embeds = self.phi.model.embed_tokens(input_ids).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
        
        inputs_embeds = torch.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], dim=1).to(self.device)  # bsz x (1+s1+1+s2) x embed_dim

        # create targets
        empty_targets = (
            torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1],  # 1 (bos) + s1 + 1
                        dtype=torch.long).to(self.device).fill_(-100)
        )  # bsz x (1 + s1)
        targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)  # bsz x (1 + s1 + 1 + s2)
        assert inputs_embeds.size()[1] == targets.size()[1]

        atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1] + 1], dtype=torch.long).to(self.device)  # bsz x (1 + s1 + 1)
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
        assert attention_mask.size() == targets.size()  # bsz x (1 + s1 + 1 + s2)

        return inputs_embeds, targets, attention_mask

    def build_prompt(self, tokenizer, captions, prompt='generate a caption'):
        input_ids, target_ids = [], []
        texts = ''
        text = '</Img> ' + prompt + '\n### Assistant: '
        texts += text
        one_input_id = tokenizer(text, add_special_tokens=False).input_ids
        input_ids += one_input_id
        target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt

        text = captions + '\n###'
        texts += text
        one_input_id = tokenizer(text, add_special_tokens=False).input_ids
        input_ids += one_input_id
        target_ids += one_input_id
        return input_ids, target_ids
    
    def proces_batch(self, tokenizer, batch_of_captions, prompt='generate a caption'):
        batch_input_ids, batch_target_ids = [], []
        for caption in batch_of_captions:
            one_input_ids, one_target_ids = self.build_prompt(tokenizer, caption, prompt)
            batch_input_ids.append(torch.LongTensor(one_input_ids))
            batch_target_ids.append(torch.LongTensor(one_target_ids))
        input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
        assert input_ids.size() == target_ids.size()
        input_ids = input_ids[:, :self.max_length]
        target_ids = target_ids[:, :self.max_length]
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        assert attention_mask.size() == input_ids.size()
        return input_ids, target_ids, attention_mask.long()

    def align(self, inputs):
        
        image_embds = self.proj(inputs['image'])
        input_ids, target_ids, attention_mask = self.proces_batch(self.tokenizer, inputs['captions'])
        inputs_embeds, targets, attention_mask = self.prompt_wrap(image_embds, input_ids, target_ids, attention_mask)
        
        inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16)
        targets = targets.long()
        attention_mask = attention_mask.to(dtype=torch.bfloat16)

        with autocast():
            outputs = self.phi(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                labels=targets,
            )

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc

    def forward(self, inputs):
        loss = 0
        gen_acc = 0
        mse_loss = None

        inputs['image'] = inputs['image'].requires_grad_()        
        if self.stage == 1:
            loss, gen_acc = self.align(inputs)
        else:
            loss, gen_acc, mse_loss = self.instruction_tuning(inputs)

        return loss, gen_acc, mse_loss