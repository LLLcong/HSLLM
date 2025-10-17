import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, LlamaModel, LlamaTokenizer
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class HSLLM(nn.Module):
    
    def __init__(self, configs, device):
        super(HSLLM, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.is_gpt = configs.is_gpt

        if configs.is_gpt:
            if configs.pretrain:
                # self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                self.llm_model = GPT2Model.from_pretrained(
                    '/home/lc/TimeSeries/pretrain/openai_gpt2/', 
                    local_files_only=True, output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    '/home/lc/TimeSeries/pretrain/openai_gpt2/',
                    local_files_only=True)
            else:
                print("------------------no pretrain------------------")
                self.llm_model = GPT2Model(GPT2Config())
            self.llm_model.h = self.llm_model.h[:configs.gpt_layers]
            # print("gpt2 = {}".format(self.llm_model))

        if configs.is_gpt:        
            if configs.freeze and configs.pretrain:
                for i, (name, param) in enumerate(self.llm_model.named_parameters()):
                    if 'ln' in name or 'wpe' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        self.top_k = 5

        self.d_model = configs.d_model
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        if configs.prompt == 'none':
            self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len) 
        else:
            self.out_layer = nn.Linear(configs.d_model * (self.patch_num+100), configs.pred_len)

        if configs.prompt == 'learnable_zero':
            self.learnable_padding = nn.Parameter(torch.zeros(1, 100, configs.d_model))  # (1, padding_len, d_model)
        elif configs.prompt == 'learnable_uniform':
            self.learnable_padding = nn.Parameter(torch.randn(1, 100, configs.d_model))
            nn.init.uniform_(self.learnable_padding)
        else:
            pass

        if configs.is_gpt:
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = 100
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.conv_layer1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=10, stride=8, padding=1)
        self.conv_layer2 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=14, stride=8, padding=3)
        
        embed_size=768
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=8)
        self.conv_proj = nn.Linear(16, 768)

        self.rnn = nn.GRU(self.patch_size, 768, batch_first=True)

        for layer in (self.llm_model, self.in_layer, self.out_layer, self.conv_layer1, self.conv_layer2, self.mapping_layer, self.rnn,
                    self.conv_proj, self.multihead_attention):
            layer.to(device=device)
            layer.train()        

    def forward(self, args, x, itr):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        if args.prompt == 'statistics':
            min_values = torch.min(x, dim=1)[0]
            max_values = torch.max(x, dim=1)[0]
            medians = torch.median(x, dim=1).values
            lags = self.calcute_lags(x)
            trends = x.diff(dim=1).sum(dim=1)

            prompt = []
            for b in range(x.shape[0]):
                min_values_str = str(min_values[b].tolist()[0])
                max_values_str = str(max_values[b].tolist()[0])
                median_values_str = str(medians[b].tolist()[0])
                lags_values_str = str(lags[b].tolist())
                prompt_ = (
                    f"<|start_prompt|>Forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                    "Input statistics: "
                    f"min value {min_values_str}, "
                    f"max value {max_values_str}, "
                    f"median value {median_values_str}, "
                    f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                    f"top 5 lags are : {lags_values_str}<|end_prompt|>"
                )
                prompt.append(prompt_)
            if self.is_gpt:
                # prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids #[256,93]
                prompt = self.tokenizer(prompt, return_tensors="pt", padding='max_length', truncation=True, max_length=100).input_ids #[256,93]
                prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x.device)) #[256,93,768]

        elif args.prompt == 'learnable_zero' or args.prompt == 'learnable_uniform':
            prompt_embeddings = self.learnable_padding.to(x.device).expand(B, -1, -1)  # Expanding padding to match batch size

        elif args.prompt == 'word_embedding' or args.prompt == 'HS_embedding' or args.prompt == 'no_conv': 
            # word embeddings
            source_embeddings = self.mapping_layer(self.word_embeddings.to(x.device).permute(1, 0)).permute(1, 0)  # (1000, 768)
            prompt_embeddings = source_embeddings.unsqueeze(0).expand(B, -1, -1)  # (B, 100, 768)                                                               
        else: # none
            pass
            

        # patch
        x_patch = rearrange(x, 'b l m -> b m l') 
        x_patch = self.padding_patch_layer(x_patch) 
        x_patch = x_patch.unfold(dimension=-1, size=self.patch_size, step=self.stride) 
        x_patch = rearrange(x_patch, 'b m n p -> (b m) n p') 

        # conv1d
        conv_output1 = self.conv_layer1(x.transpose(1, 2))   
        conv_output1 = conv_output1.transpose(1, 2) 
        conv_output2 = self.conv_layer2(x.transpose(1, 2))   
        conv_output2 = conv_output2.transpose(1, 2) 
        conv_output = conv_output1 + conv_output2  
        conv_output_proj = self.conv_proj(conv_output)  


        # GRU token
        if args.prompt == 'no_conv':
            _, hidden_state = self.rnn(x_patch) 
        else:
            _, hidden_state = self.rnn(x_patch + conv_output) 
        # global_token = hidden_state.permute(1, 0, 2)  

        if args.prompt == 'HS_embedding' or args.prompt == 'no_conv':
            # Multihead Attention, source_embeddings - Q, hidden_state - K, V, output_attentioned.shape=[100, 128, 768]
            prompt_embeddings = prompt_embeddings.permute(1, 0, 2)  # [100, batch_size, 768]
            output_attentioned, attention_weights = self.multihead_attention(prompt_embeddings, hidden_state, hidden_state)
            prompt_embeddings = output_attentioned.permute(1, 0, 2)


        outputs = self.in_layer(x_patch)
        outputs = outputs + conv_output_proj

        if args.prompt != 'none':
            outputs = torch.concat([prompt_embeddings, outputs], dim=1)  


        if self.is_gpt:
            outputs = self.llm_model(inputs_embeds=outputs).last_hidden_state

        outputs = outputs.reshape(B*M, -1)
        outputs = self.out_layer(outputs) 
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags