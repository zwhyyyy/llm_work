import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from typing import List
import xml.etree.ElementTree as ET


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.PAD_TOKEN = '<pad>'
        self.SOS_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        self.UNK_TOKEN = '<unk>'
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.SOS_TOKEN)
        self.add_word(self.EOS_TOKEN)
        self.add_word(self.UNK_TOKEN)
        self.pad_idx = self.word2idx[self.PAD_TOKEN]
        self.sos_idx = self.word2idx[self.SOS_TOKEN]
        self.eos_idx = self.word2idx[self.EOS_TOKEN]
        self.unk_idx = self.word2idx[self.UNK_TOKEN]
    
    def add_word(self, word: str):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, sentence: List[str]) -> List[int]:
        return [self.word2idx.get(word, self.unk_idx) for word in sentence]
    
    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices]


def parse_xml_file(file_path: str) -> List[str]:
    sentences = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        for seg in root.iter('seg'):
            if seg.text:
                sentences.append(seg.text.strip())
    except:
        print(f"无法解析XML文件: {file_path}")
    return sentences


def parse_train_file(file_path: str) -> List[str]:
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('<'):
                sentences.append(line)
    return sentences


def tokenize(sentence: str) -> List[str]:
    sentence = re.sub(r'([,.!?;:])', r' \1 ', sentence)
    tokens = [token.lower() for token in sentence.split() if token.strip()]
    return tokens


def build_vocabulary(sentences: List[str], max_vocab_size: int = 30000, min_freq: int = 2) -> Vocabulary:
    vocab = Vocabulary()
    word_freq = Counter()
    for sentence in sentences:
        tokens = tokenize(sentence)
        word_freq.update(tokens)
    for word, freq in word_freq.most_common(max_vocab_size):
        if freq >= min_freq:
            vocab.add_word(word)
    print(f"词汇表大小: {len(vocab)}")
    return vocab


class TranslationDataset(Dataset):
    def __init__(self, src_sentences: List[str], tgt_sentences: List[str], 
                 src_vocab: Vocabulary, tgt_vocab: Vocabulary, max_len: int = 100):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        filtered_data = []
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_tokens = tokenize(src)
            tgt_tokens = tokenize(tgt)
            if len(src_tokens) <= max_len and len(tgt_tokens) <= max_len:
                filtered_data.append((src, tgt))
        self.src_sentences = [item[0] for item in filtered_data]
        self.tgt_sentences = [item[1] for item in filtered_data]
        print(f"过滤后的数据集大小: {len(self.src_sentences)}")
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_tokens = tokenize(self.src_sentences[idx])
        tgt_tokens = tokenize(self.tgt_sentences[idx])
        src_indices = self.src_vocab.encode(src_tokens)
        tgt_indices = self.tgt_vocab.encode(tgt_tokens)
        src_indices.append(self.src_vocab.eos_idx)
        tgt_indices.append(self.tgt_vocab.eos_idx)
        return torch.tensor(src_indices, dtype=torch.long), torch.tensor(tgt_indices, dtype=torch.long)


def collate_fn(batch, src_pad_idx: int, tgt_pad_idx: int):
    src_batch, tgt_batch = zip(*batch)
    src_max_len = max(len(s) for s in src_batch)
    tgt_max_len = max(len(t) for t in tgt_batch)
    src_padded = []
    tgt_padded = []
    for src, tgt in zip(src_batch, tgt_batch):
        src_pad_len = src_max_len - len(src)
        tgt_pad_len = tgt_max_len - len(tgt)
        src_padded.append(torch.cat([src, torch.tensor([src_pad_idx] * src_pad_len, dtype=torch.long)]))
        tgt_padded.append(torch.cat([tgt, torch.tensor([tgt_pad_idx] * tgt_pad_len, dtype=torch.long)]))
    return torch.stack(src_padded), torch.stack(tgt_padded)


def load_data(train_src_path: str, train_tgt_path: str, 
              dev_src_path: str, dev_tgt_path: str,
              batch_size: int = 32, max_len: int = 100):
    print("加载训练数据...")
    train_src_sentences = parse_train_file(train_src_path)
    train_tgt_sentences = parse_train_file(train_tgt_path)
    print(f"训练集大小: {len(train_src_sentences)}")
    print("加载验证数据...")
    dev_src_sentences = parse_xml_file(dev_src_path)
    dev_tgt_sentences = parse_xml_file(dev_tgt_path)
    print(f"验证集大小: {len(dev_src_sentences)}")
    print("构建词汇表...")
    src_vocab = build_vocabulary(train_src_sentences, max_vocab_size=30000)
    tgt_vocab = build_vocabulary(train_tgt_sentences, max_vocab_size=30000)
    print("创建数据集...")
    train_dataset = TranslationDataset(train_src_sentences, train_tgt_sentences, 
                                      src_vocab, tgt_vocab, max_len)
    dev_dataset = TranslationDataset(dev_src_sentences, dev_tgt_sentences, 
                                    src_vocab, tgt_vocab, max_len)
    print("创建数据加载器...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, src_vocab.pad_idx, tgt_vocab.pad_idx),
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, src_vocab.pad_idx, tgt_vocab.pad_idx),
        num_workers=0
    )
    return train_loader, dev_loader, src_vocab, tgt_vocab
