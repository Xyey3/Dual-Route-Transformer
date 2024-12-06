import os
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


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
