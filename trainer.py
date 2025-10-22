import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class LearningRateScheduler:
    def __init__(self, optimizer, initial_lr: float = 0.001, min_lr: float = 0.00001, total_steps: int = 10000):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.current_step = 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = initial_lr
    
    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        if self.current_step >= self.total_steps:
            return self.min_lr
        decay_ratio = self.current_step / self.total_steps
        lr = self.initial_lr - (self.initial_lr - self.min_lr) * decay_ratio
        return max(lr, self.min_lr)
    
    def get_current_lr(self):
        return self.optimizer.param_groups[0]['lr']


class Trainer:
    def __init__(self, model, train_loader, dev_loader, src_pad_idx: int, tgt_pad_idx: int,
                 device: torch.device, config: Dict):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        self.config = config
        self.criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
        initial_lr = config.get('initial_lr', 0.001)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=config.get('weight_decay', 0.0001)
        )
        total_epochs = config.get('num_epochs', 100)
        steps_per_epoch = len(train_loader)
        total_steps = total_epochs * steps_per_epoch
        self.scheduler = LearningRateScheduler(
            self.optimizer,
            initial_lr=config.get('initial_lr', 0.001),
            min_lr=config.get('min_lr', 0.00001),
            total_steps=total_steps
        )
        self.max_grad_norm = config.get('max_grad_norm', 1.0)
        self.history = {
            'train_loss': [],
            'dev_loss': [],
            'train_ppl': [],
            'dev_ppl': [],
            'lr': [],
            'batch_train_loss': [],
            'batch_dev_loss': [],
            'batch_steps': [],
            'batch_epoch': []
        }
        self.best_dev_loss = float('inf')
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        batch_count = 0
        global_step = (epoch - 1) * len(self.train_loader)
        start_time = time.time()
        
        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = self.model(src, tgt_input)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            loss = self.criterion(output, tgt_output)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / batch_count
                current_lr = self.scheduler.get_current_lr()
                elapsed = time.time() - start_time
                dev_loss, _ = self.evaluate()
                self.history['batch_train_loss'].append(avg_loss)
                self.history['batch_dev_loss'].append(dev_loss)
                self.history['batch_steps'].append(global_step + batch_idx + 1)
                self.history['batch_epoch'].append(epoch)
                self.model.train()
                print(f'Epoch [{epoch}], Batch [{batch_idx + 1}/{len(self.train_loader)}], '
                      f'Train Loss: {avg_loss:.4f}, Dev Loss: {dev_loss:.4f}, '
                      f'LR: {current_lr:.6f}, Time: {elapsed:.2f}s')
        
        avg_loss = total_loss / batch_count
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 100 else float('inf')
        current_lr = self.scheduler.get_current_lr()
        return avg_loss, avg_ppl, current_lr
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        batch_count = 0
        with torch.no_grad():
            for src, tgt in self.dev_loader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                output = self.model(src, tgt_input)
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                tgt_output = tgt_output.contiguous().view(-1)
                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()
                batch_count += 1
        avg_loss = total_loss / batch_count
        avg_ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 100 else float('inf')
        return avg_loss, avg_ppl
    
    def train(self, num_epochs: int, save_dir: str = 'checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        print("=" * 80)
        print("开始训练")
        print("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'=' * 80}")
            train_loss, train_ppl, current_lr = self.train_epoch(epoch)
            train_ppl_str = f"{train_ppl:.2f}" if train_ppl != float('inf') else "inf"
            print(f"\n训练集 - Loss: {train_loss:.4f}, PPL: {train_ppl_str}, LR: {current_lr:.6f}")
            dev_loss, dev_ppl = self.evaluate()
            dev_ppl_str = f"{dev_ppl:.2f}" if dev_ppl != float('inf') else "inf"
            print(f"验证集 - Loss: {dev_loss:.4f}, PPL: {dev_ppl_str}")
            
            if epoch > 1:
                loss_improvement = self.history['dev_loss'][-1] - dev_loss
                if loss_improvement > 0:
                    print(f"📈 验证Loss改善: {loss_improvement:.4f}")
                else:
                    print(f"📉 验证Loss变差: {abs(loss_improvement):.4f}")
            
            self.history['train_loss'].append(train_loss)
            self.history['dev_loss'].append(dev_loss)
            self.history['train_ppl'].append(train_ppl)
            self.history['dev_ppl'].append(dev_ppl)
            self.history['lr'].append(current_lr)
            
            if dev_loss < self.best_dev_loss:
                self.best_dev_loss = dev_loss
                self.save_checkpoint(os.path.join(save_dir, 'best_model.pt'), epoch)
                print(f"✓ 保存最佳模型 (dev_loss: {dev_loss:.4f})")
            
            if epoch % self.config.get('save_every', 5) == 0:
                self.save_checkpoint(os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'), epoch)
                print(f"✓ 保存检查点 (epoch {epoch})")
            
            self.plot_training_curves(save_dir)
        
        print("\n" + "=" * 80)
        print("训练完成！")
        print("=" * 80)
        print(f"最佳验证集损失: {self.best_dev_loss:.4f}")
        self._save_training_summary(save_dir)
    
    def save_checkpoint(self, path: str, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_step': self.scheduler.current_step,
            'best_dev_loss': self.best_dev_loss,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.current_step = checkpoint['scheduler_step']
        self.best_dev_loss = checkpoint['best_dev_loss']
        self.history = checkpoint['history']
        print(f"✓ 成功加载检查点 (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    
    def _save_training_summary(self, save_dir: str):
        summary_path = os.path.join(save_dir, 'training_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("训练摘要\n")
            f.write("=" * 80 + "\n\n")
            num_epochs = len(self.history['train_loss'])
            f.write(f"训练轮数: {num_epochs}\n\n")
            f.write("训练集指标:\n")
            f.write(f"  最终Loss: {self.history['train_loss'][-1]:.4f}\n")
            if self.history['train_ppl'][-1] != float('inf'):
                f.write(f"  最终PPL: {self.history['train_ppl'][-1]:.2f}\n")
            f.write(f"  最低Loss: {min(self.history['train_loss']):.4f}\n\n")
            f.write("验证集指标:\n")
            f.write(f"  最终Loss: {self.history['dev_loss'][-1]:.4f}\n")
            if self.history['dev_ppl'][-1] != float('inf'):
                f.write(f"  最终PPL: {self.history['dev_ppl'][-1]:.2f}\n")
            f.write(f"  最低Loss: {min(self.history['dev_loss']):.4f}\n")
            f.write(f"  最佳epoch: {self.history['dev_loss'].index(min(self.history['dev_loss'])) + 1}\n\n")
            f.write("学习率:\n")
            f.write(f"  最大学习率: {max(self.history['lr']):.6f}\n")
            f.write(f"  最终学习率: {self.history['lr'][-1]:.6f}\n\n")
            f.write("=" * 80 + "\n")
            f.write("逐轮详细信息\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train PPL':<12} {'Dev Loss':<12} {'Dev PPL':<12} {'LR':<12}\n")
            f.write("-" * 80 + "\n")
            for i in range(num_epochs):
                train_ppl_str = f"{self.history['train_ppl'][i]:.2f}" if self.history['train_ppl'][i] != float('inf') else "inf"
                dev_ppl_str = f"{self.history['dev_ppl'][i]:.2f}" if self.history['dev_ppl'][i] != float('inf') else "inf"
                f.write(f"{i+1:<8} "
                       f"{self.history['train_loss'][i]:<12.4f} "
                       f"{train_ppl_str:<12} "
                       f"{self.history['dev_loss'][i]:<12.4f} "
                       f"{dev_ppl_str:<12} "
                       f"{self.history['lr'][i]:<12.6f}\n")
        print(f"✓ 训练摘要已保存到: {summary_path}")
    
    def plot_training_curves(self, save_dir: str):
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=6)
        line2 = ax1.plot(epochs, self.history['dev_loss'], 'r-', label='Dev Loss', linewidth=2, marker='s', markersize=6)
        ax1.set_xlabel('Epoch', fontsize=13)
        ax1.set_ylabel('Loss', fontsize=13, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(True, alpha=0.3)
        line3 = ax1_twin.plot(epochs, self.history['lr'], 'g--', label='Learning Rate', linewidth=2, marker='d', markersize=5)
        ax1_twin.set_ylabel('Learning Rate', fontsize=13, color='green')
        ax1_twin.tick_params(axis='y', labelcolor='green')
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', fontsize=11)
        ax1.set_title('Training/Validation Loss and Learning Rate', fontsize=14, fontweight='bold')
        
        if len(self.history['batch_steps']) > 0:
            epoch1_indices = [i for i, e in enumerate(self.history['batch_epoch']) if e == 1]
            if epoch1_indices:
                epoch1_steps = [self.history['batch_steps'][i] for i in epoch1_indices]
                epoch1_train_loss = [self.history['batch_train_loss'][i] for i in epoch1_indices]
                epoch1_dev_loss = [self.history['batch_dev_loss'][i] for i in epoch1_indices]
                axes[1].plot(epoch1_steps, epoch1_train_loss, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
                axes[1].plot(epoch1_steps, epoch1_dev_loss, 'r-', label='Dev Loss', linewidth=2, marker='s', markersize=4)
                axes[1].set_xlabel('Training Steps (Epoch 1)', fontsize=13)
                axes[1].set_ylabel('Loss', fontsize=13)
                axes[1].set_title('First Epoch Detailed Loss (every 100 batches)', fontsize=14, fontweight='bold')
                axes[1].legend(fontsize=11)
                axes[1].grid(True, alpha=0.3)
        
        train_ppl = [(i+1, p) for i, p in enumerate(self.history['train_ppl']) if p != float('inf')]
        dev_ppl = [(i+1, p) for i, p in enumerate(self.history['dev_ppl']) if p != float('inf')]
        if train_ppl:
            train_epochs, train_ppl_values = zip(*train_ppl)
            axes[2].plot(train_epochs, train_ppl_values, 'b-', label='Train PPL', linewidth=2, marker='o', markersize=6)
        if dev_ppl:
            dev_epochs, dev_ppl_values = zip(*dev_ppl)
            axes[2].plot(dev_epochs, dev_ppl_values, 'r-', label='Dev PPL', linewidth=2, marker='s', markersize=6)
        axes[2].set_xlabel('Epoch', fontsize=13)
        axes[2].set_ylabel('Perplexity', fontsize=13)
        axes[2].set_title('Training/Validation Perplexity', fontsize=14, fontweight='bold')
        axes[2].legend(fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'training_curves.pdf'), bbox_inches='tight')
        plt.close()
