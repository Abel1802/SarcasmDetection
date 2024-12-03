import os
import glob
import json
import numpy as np
from tqdm import tqdm
import soundfile as sf
import torch
from sentence_transformers import SentenceTransformer
from transformers import Wav2Vec2Processor, Wav2Vec2Model


def load_ssm_model():
    '''
        load a pre-trained speech large model.
    '''
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    return model, processor


def process_a_audio_file(model, processor, audio_path):
    '''
        process a audio file.
    '''
    # 2. 读取音频文件
    # audio_path = "/projects/0/prjs0864/tts/raw_data/podcast/all_processed/overly-sarcastic-podcast__ospod-episode-83-tone-arm/overly-sarcastic-podcast__ospod-episode-83-tone-arm_0.mp3"  # 替换为你的音频文件路径
    waveform, sample_rate = sf.read(audio_path)

    # 确保采样率为16kHz（与模型要求一致）
    if sample_rate != 16000:
        from scipy.signal import resample
        waveform = resample(waveform, int(len(waveform) * 16000 / sample_rate))
        sample_rate = 16000

    # 如果是多通道音频，取第一个通道
    if len(waveform.shape) > 1:
        waveform = waveform[:, 0]

    # 3. 处理音频数据
    input_values = processor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_values

    # 4. 提取特征
    with torch.no_grad():
        features = model(input_values).last_hidden_state.squeeze(0)  # [batch_size, sequence_length, hidden_size]

    # 5. 打印特征信息
    # print("Features shape:", features.shape)  # 示例: torch.Size([1, 2000, 768])
    return features


def extract_audio_data(data_dir):
    '''
        load audio data from a directory.
    '''
    # load a pre-trained speech large model.
    model, processor = load_ssm_model()

    audio_dir = '/projects/0/prjs0864/tts/raw_data/podcast/all_processed'
    print(f'Loading data from{data_dir}...')
    audio_embs = []
    labels = []
    for file in tqdm(os.listdir(data_dir), desc="Processing files", unit="file"):
        if file.endswith('.json'):
            f = open(os.path.join(data_dir, file), 'r')
            raw_data = json.load(f)
            for item in tqdm(raw_data, desc=f"Processing items in {file}", leave=False, unit="item"):
            # for item in raw_data:
                text = item['text']
                sarcasm = item['sarcasm']
                idx = item['index']
                episode_name = file.split('.')[0]
                tmp_audio_path = os.path.join(audio_dir, f'overly-sarcastic-podcast__ospod-episode-{episode_name}')
                audio_path = os.path.join(tmp_audio_path, f'ospod-episode-{episode_name}_{idx}.mp3')
                # print(int(sarcasm), audio_path)
                audio_emb = process_a_audio_file(model, processor, audio_path)
                audio_embs.append(audio_emb)
                labels.append(int(sarcasm))
    data = {"audio_embs": audio_embs, "labels": labels}
    save_path = f'./processed_data/wav2vec_{audio_emb.shape[-1]}_labels.pt'
    torch.save(data, save_path)
    print(f'Embeddings saved to {save_path}')


def load_bert_model(model_name='all-mpnet-base-v2'):
    '''
    Load a pre-trained BERT model from the Hugging Face model hub.
    Args: 
        model_name: 
            1. all-MiniLM-L6-v2 (384,)
            2. all-mpnet-base-v2 (768,)
            3. all-roberta-large-v1 (1024,)
    '''
    model = SentenceTransformer(model_name)
    
    # sentence = 'This is a sample sentence.'
    # embedding = model.encode(sentence)
    # print(f"Embedding shape: {embedding.shape}")
    return model


def load_data(data_dir):
    print(f'Loading data from{data_dir}...')
    data = []
    label = []
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            f = open(os.path.join(data_dir, file), 'r')
            raw_data = json.load(f)
            for item in raw_data:
                text = item['text']
                sarcasm = item['sarcasm']
                # print(int(sarcasm), text)
                data.append(text)
                label.append(int(sarcasm))
    return data, label


def get_bert_embeddings(data, labels, model, max_length=100):
    print('Getting BERT embeddings...')
    embeddings = []
    for text in tqdm(data):
        # text = text[:max_length]
        model_output = model.encode(text)
        embeddings.append(model_output)
    np.savez(f'./processed_data/bert_embeddings_{model_output.shape[0]}_labels', array1=np.array(embeddings), array2=np.array(labels))
    print(f'Embeddings saved to processed_data/bert_embeddings_{model_output.shape[0]}_labels')


def process_text():
    data, label = load_data('/projects/0/prjs0864/tts/raw_data/podcast/chatgpt/all_jsons')
    # model = load_bert_model(model_name='all-MiniLM-L6-v2')
    # model = load_bert_model(model_name='all-mpnet-base-v2')
    model = load_bert_model(model_name='all-roberta-large-v1')
    get_bert_embeddings(data, label, model)


def main():
    # model, processor = load_ssm_model()
    # process_a_audio_file(model, processor)
    extract_audio_data(data_dir='/projects/0/prjs0864/tts/raw_data/podcast/chatgpt/all_jsons')


if __name__ == '__main__':
    main()