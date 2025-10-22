import os
import torch
import random
import numpy as np
from model import Transformer
from data_process import load_data
from trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)
    
    config = {
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'd_ff': 2048,
        'max_len': 150,
        'dropout': 0.1,
        'batch_size': 64,
        'num_epochs': 10,
        'initial_lr': 0.001,
        'min_lr': 0.00001,
        'max_grad_norm': 1.0,
        'weight_decay': 0.0001,
        'save_every': 1,
        'max_sentence_len': 100,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    train_src_path = 'en-de/train.tags.en-de.en'
    train_tgt_path = 'en-de/train.tags.en-de.de'
    dev_src_path = 'en-de/IWSLT17.TED.dev2010.en-de.en.xml'
    dev_tgt_path = 'en-de/IWSLT17.TED.dev2010.en-de.de.xml'
    
    print("\n" + "=" * 80)
    print("数据加载")
    print("=" * 80)
    train_loader, dev_loader, src_vocab, tgt_vocab = load_data(
        train_src_path, train_tgt_path,
        dev_src_path, dev_tgt_path,
        batch_size=config['batch_size'],
        max_len=config['max_sentence_len']
    )
    
    print(f"\n源语言词汇表大小: {len(src_vocab)}")
    print(f"目标语言词汇表大小: {len(tgt_vocab)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(dev_loader)}")
    
    print("\n" + "=" * 80)
    print("创建Transformer模型")
    print("=" * 80)
    
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_len=config['max_len'],
        dropout=config['dropout'],
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx
    )
    
    print("\n模型参数统计:")
    total_params, trainable_params = model.count_parameters()
    print(f"\n模型大小: {total_params / 1e6:.2f}M 参数")
    
    print("\n" + "=" * 80)
    print("训练器初始化")
    print("=" * 80)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
        device=device,
        config=config
    )
    
    print(f"优化器: AdamW")
    print(f"学习率调度: 线性衰减 ({config['initial_lr']} → {config['min_lr']})")
    print(f"训练轮数: {config['num_epochs']} epochs")
    print(f"梯度裁剪: max_norm={config['max_grad_norm']}")
    print(f"权重衰减: {config['weight_decay']}")
    
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    
    trainer.train(
        num_epochs=config['num_epochs'],
        save_dir=save_dir
    )
    
    print("\n保存词汇表...")
    torch.save({
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'config': config
    }, os.path.join(save_dir, 'vocab_and_config.pt'))
    print("✓ 词汇表保存完成")
    
    print("\n" + "=" * 80)
    print("全部完成！")
    print("=" * 80)
    print(f"模型检查点保存在: {save_dir}/")
    print(f"训练曲线保存在: {save_dir}/training_curves.png")


if __name__ == '__main__':
    main()
