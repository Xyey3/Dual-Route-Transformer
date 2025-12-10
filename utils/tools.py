import os
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # safe getters
    warmup_epochs = getattr(args, 'warmup_epochs', 5)
    total_epochs = getattr(args, 'train_epochs', 50)
    decay_interval = getattr(args, 'decay_interval', 5)
    interval = getattr(args, 'type1_interval', 1)
    
    # 1) warmup only (linear warmup to args.learning_rate)
    if args.lradj == 'warmup':
        if epoch <= warmup_epochs:
            lr = args.learning_rate * epoch / max(1, warmup_epochs)
        else:
            lr = args.learning_rate
        for g in optimizer.param_groups:
            g['lr'] = lr
        print(f'Warmup LR -> {lr:.6g}')
        return

    # 2) warmup + cosine (useful)
    if args.lradj == 'warmup_cosine' or args.lradj == 'cosine':
        # linear warmup then cosine decay
        if epoch <= warmup_epochs:
            lr = args.learning_rate * epoch / max(1, warmup_epochs)
        else:
            # cosine decay from epoch = warmup_epochs..total_epochs
            t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
            lr = 0.5 * args.learning_rate * (1 + math.cos(math.pi * t))
        for g in optimizer.param_groups:
            g['lr'] = lr
        print(f'Warmup+Cosine LR -> {lr:.6g}')
        return

    # 3) slow decay (halve every n epochs)
    if args.lradj == 'slow':
        decay_times = epoch // decay_interval
        lr = args.learning_rate * (0.5 ** decay_times)
        for g in optimizer.param_groups:
            g['lr'] = lr
        print(f'Slow Decay LR -> {lr:.6g}')
        return

    # 4) improved type1: decay every x epochs instead of every epoch
    if args.lradj == 'type1':
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // interval))
        for g in optimizer.param_groups:
            g['lr'] = lr
        print(f'Type1 LR -> {lr:.6g}')
        return

    # 5) constant / none
    if args.lradj == 'constant' or args.lradj == 'none':
        for g in optimizer.param_groups:
            g['lr'] = args.learning_rate
        return

    # fallback
    for g in optimizer.param_groups:
        g['lr'] = args.learning_rate
    print(f'Fallback const LR -> {args.learning_rate:.6g}')
    
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.last_checkpoint = None

    def __call__(self, val_loss, model, path):
        """Check validation loss and update checkpoint state.
        Behavior:
        - Save model when validation loss improves.
        - Increase counter when validation loss does not improve.
        - Trigger early stop when counter >= patience and then restore the best saved model.
        """
        score = -val_loss

        # On first observed validation, save and initialize best score.
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            return

        # Validation loss did not improve.
        if score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            # Trigger early stopping when patience exceeded: restore best model once.
            if self.counter >= self.patience:
                print("Early stopping triggered. Restoring best model...")
                self.load_checkpoint(model)
                self.early_stop = True

        # Validation loss improved: save checkpoint and reset counter.
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        """Save model state_dict to 'checkpoint.pth' under the given path."""
        # Ensure the directory exists.
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")

        # Construct checkpoint file path and save only the model weights.
        checkpoint_path = os.path.join(path, "checkpoint.pth")
        torch.save(model.state_dict(), checkpoint_path)

        # Update tracking variables.
        self.val_loss_min = val_loss
        self.last_checkpoint = checkpoint_path

    def load_checkpoint(self, model):
        """Load the best saved model weights into the provided model."""
        if self.last_checkpoint is None or not os.path.exists(self.last_checkpoint):
            print("No checkpoint found to restore.")
            return

        # Load onto the device where the model currently lives.
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

        checkpoint = torch.load(self.last_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        print("Best model restored.")


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def visualize_attention(layers_attns, save_path="multi_attention_visualization.pdf"):
    """
    可视化多个层的注意力矩阵并保存为单独的 PDF 文件。
    随机选择一个样本进行可视化。

    参数：
    - layers_attns: 列表，每个元素是一个元组，包含多个注意力矩阵，每个注意力矩阵形状分别为 (num_variates, num_heads, seg_num, seg_num), (seg_num, num_heads, num_variates, num_variates)。
    - save_path: 保存 PDF 文件的路径。
    """
    sample_idx1 = random.randint(0, layers_attns[0][0].shape[0] - 1)
    sample_idx2 = random.randint(0, layers_attns[0][0].shape[2] - 1)
    # sample_idx1 = 0
    # sample_idx2 = 2
    for layer_idx, attns in enumerate(layers_attns):  # 遍历每一层
        for attn_idx, attn in enumerate(attns):  # 遍历每层中的每个 attention
            _, num_heads, _, _ = attn.shape
            if attn_idx == 0:
                attn_sample = attn[sample_idx1]
            else:
                attn_sample = attn[sample_idx2]

            k=1
            # 定义子图布局为两行
            ncols = math.ceil(num_heads / k)  # 每行的列数
            fig = plt.figure(figsize=(6 * ncols, 9))
            gs = fig.add_gridspec(k, ncols + 1, width_ratios=[1] * ncols + [0.1])  # 添加显色棒一列
            
            # fig.suptitle(f"Layer {layer_idx + 1}, Attention {attn_idx + 1}, Random Sample {sample_idx + 1}", fontsize=16)

            vmin, vmax = attn_sample.min().item(), attn_sample.max().item()
            axes = []

            for row in range(k):
                for col in range(ncols):
                    if len(axes) < num_heads:  # 限制为 num_heads 个图
                        ax = fig.add_subplot(gs[row, col])
                        axes.append(ax)

            for head_idx, ax in enumerate(axes):
                cax = ax.matshow(attn_sample[head_idx].cpu(), cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(f"Head {head_idx + 1}", fontsize=12)
                ax.set_xlabel("Key Position")
                ax.set_ylabel("Query Position")

            # 添加显色棒
            cbar_ax = fig.add_subplot(gs[:, -1])  # 全图共享显色棒
            fig.colorbar(cax, cax=cbar_ax, orientation="vertical")

            # 保存当前图到 PDF 文件中
            if attn_idx == 0:
                save_path_layer = save_path.replace(".pdf", f"_layer{layer_idx + 1}_attn{attn_idx + 1}_variate{sample_idx1}.pdf")
            else:
                save_path_layer = save_path.replace(".pdf", f"_layer{layer_idx + 1}_attn{attn_idx + 1}_seg{sample_idx2}.pdf")
            plt.savefig(save_path_layer, format="pdf", bbox_inches="tight")
            plt.close(fig)

    print(f"All attention visualizations saved as individual PDF files with base name '{save_path}'")



def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
