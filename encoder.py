import json
import os
import pickle
from time import time

import auto_gptq
import clip
import numpy as np
import openai
import torch
import torchvision.transforms as T
from auto_gptq.modeling import BaseGPTQForCausalLM
from openai import OpenAI
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
    layers_block_name = "model.layers"
    outside_layer_modules = [
        'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output',
    ]
    inside_layer_modules = [
        ["attention.wqkv.linear"],
        ["attention.wo.linear"],
        ["feed_forward.w1.linear", "feed_forward.w3.linear"],
        ["feed_forward.w2.linear"],
    ]

def load_xcomposer():
    auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
    torch.set_grad_enabled(False)
    model = InternLMXComposer2QForCausalLM.from_quantized('internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True, device="cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b-4bit', trust_remote_code=True)
    return model, tokenizer


sentence_models = ['text-embedding-ada-002', 'text-embedding-3-large', 'all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'clip']


def encode_sentences(sentence_list, model_name):
    '''given a list of sentences, return the embeddings for them using the sentence encoder model'''
    assert model_name in sentence_models
    emb_list = []
    if model_name in['text-embedding-ada-002', 'text-embedding-3-large']: #openai embedding requires api-key
        client = OpenAI()
        emb = client.embeddings.create(input=sentence_list, model=model_name)
        for i in range(len(sentence_list)):
            emb_list.append(np.array(emb.data[i].embedding).reshape(1, -1))
        emb_list = np.concatenate(emb_list, axis=0)
        return emb_list
    elif model_name == 'clip': # clip embedding
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, transform = clip.load("ViT-B/32", device=device)
        with torch.no_grad():
            for sentence in sentence_list:
                emb_list.append(model.encode_text(clip.tokenize([sentence]).to(device)).cpu().numpy())
        emb_list = np.concatenate(emb_list, axis=0)
        return emb_list
    else: #sentence transformer encoder
        model = SentenceTransformer('sentence-transformers/'+model_name)
        num = len(sentence_list)
        batch_size = 10
        batch_num = num // batch_size
        with torch.no_grad():
            for batch_id in range(batch_num):
                batch_sentences = sentence_list[batch_id*10: (batch_id+1)*10]
                emb_list.append(model.encode(batch_sentences))
            if batch_num * 10 < num: #remaining <10 sentences
                remaining_sentences = sentence_list[batch_num*10: num]
                emb_list.append(model.encode(remaining_sentences))
        return emb_list


if __name__ == '__main__':
    encode_sentences(['hello!', 'what'], model_name='text-embedding-ada-002')
