import argparse
import os
import sys
import wandb
import subprocess
from transformers import BertConfig
import numpy as np
import logging
import torch
import torch.nn as nn
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    recall_score,
    precision_score,
)

# 下面是自定义的库的引入
from PhosphoricAcidDetect.PAdetect.training_utils import (
    LargeCRFPartitionDataset,
    RegionCRFDataset,
    SIGNALP6_GLOBAL_LABEL_DICT,
    SIGNALP_KINGDOM_DICT,
    PhosphoicAcidThreeLineFastaDataset,
    compute_cosine_region_regularization,
)
from PhosphoricAcidDetect.PAdetect.models import ProteinBertTokenizer, BertSequenceTaggingCRF
from PhosphoricAcidDetect.PAdetect.utils import class_aware_cosine_similarities, get_region_lengths


# 这边用wandb进行一些数据的记录
# 如果移植服务器出现问题，把这部分删除掉
def log_metrics(metrics_dict, split: str, step: int):
    """Convenience function to add prefix to all metrics before logging."""
    wandb.log(
        {
            f"{split.capitalize()} {name.capitalize()}": value
            for name, value in metrics_dict.items()
        },
        step=step,
    )


# This is a quick fix for hyperparameter search.
# wandb reinit does not work on scientific linux yet, so use
# a pseudo-wandb instead of the actual wandb library
# 还没尝试wandb会在linux上出现什么问题，目前先保留

class DecoyConfig:
    def update(*args, **kwargs):
        pass


class DecoyWandb:
    config = DecoyConfig()

    def init(self, *args, **kwargs):
        print(
            "Decoy Wandb initiated, override wandb with no-op logging to prevent errors."
        )
        pass

    def log(self, value_dict, *args, **kwargs):
        # filter for train logs here, don't want to print at every step
        if list(value_dict.keys())[0].startswith("Train"):
            pass
        else:
            print(value_dict)
            print(args)
            print(kwargs)

    def watch(self, *args, **kwargs):
        pass


# get the git hash - and log it
# wandb does that automatically - but only when in the correct directory when launching the job.
# by also doing it manually, force to launch from the correct directory, because otherwise this command will fail.
GIT_HASH = (
    subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode()
)

MODEL_DICT = {
    "bert_prottrans": (BertConfig, BertSequenceTaggingCRF),
}

TOKENIZER_DICT = {
    "bert_prottrans": (ProteinBertTokenizer, "Rostlab/prot_bert")
}


def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    c_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    return logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 将标记序列转换为切割位点的索引
# 实际上不需要进行切割操作，这部分代码暂且不考虑
def tagged_seq_to_cs_multiclass(tagged_seqs: np.ndarray, sp_tokens=[0, 4, 5]):
    """Convert a sequences of tokens to the index of the cleavage site.
    Inputs:
        tagged_seqs: (batch_size, seq_len) integer array of position-wise labels
        sp_tokens: label tokens that indicate a signal peptide
    Returns:
        cs_sites: (batch_size) integer array of last position that is a SP. NaN if no SP present in sequence.
    """

    # 获得最后一个信号肽的位置
    def get_last_sp_idx(x: np.ndarray) -> int:
        """Func1d to get the last index that is tagged as SP. use with np.apply_along_axis. """
        sp_idx = np.where(np.isin(x, sp_tokens))[0]
        if len(sp_idx) > 0:
            max_idx = sp_idx.max() + 1
        else:
            max_idx = np.nan
        return max_idx

    cs_sites = np.apply_along_axis(get_last_sp_idx, 1, tagged_seqs)
    return cs_sites


# 用于从模型输出中获取评估指标，集中与切割位点与全局标签上，目前保留但不予使用
def report_metrics(
        true_global_labels: np.ndarray,
        pred_global_labels: np.ndarray,
        true_sequence_labels: np.ndarray,
        pred_sequence_labels: np.ndarray,
        use_cs_tag=False,
) -> Dict[str, float]:
    # 输入参数依次为：
    # true_global_labels: 真实的全局标签。
    # pred_global_labels: 预测的全局标签。
    # true_sequence_labels: 真实的序列标签。
    # pred_sequence_labels: 预测的序列标签。
    # use_cs_tag: 是否使用特定的切割位点标签，我们不使用，可以不修改，默认设置为False即可
    """Utility function to get metrics from model output"""

    # 这部分代码处理切割位点，不会被执行
    true_cs = tagged_seq_to_cs_multiclass(
        true_sequence_labels, sp_tokens=[4, 9, 14] if use_cs_tag else [3, 7, 11]
    )
    pred_cs = tagged_seq_to_cs_multiclass(
        pred_sequence_labels, sp_tokens=[4, 9, 14] if use_cs_tag else [3, 7, 11]
    )
    pred_cs = pred_cs[~np.isnan(true_cs)]
    true_cs = true_cs[~np.isnan(true_cs)]
    true_cs[np.isnan(true_cs)] = -1
    pred_cs[np.isnan(pred_cs)] = -1

    # applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent to the argmax.
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)
    metrics_dict = {}
    metrics_dict["CS Recall"] = recall_score(true_cs, pred_cs, average="micro")
    metrics_dict["CS Precision"] = precision_score(true_cs, pred_cs, average="micro")
    metrics_dict["CS MCC"] = matthews_corrcoef(true_cs, pred_cs)
    metrics_dict["Detection MCC"] = matthews_corrcoef(
        true_global_labels, pred_global_labels_thresholded
    )

    return metrics_dict


# 存在界id的情况下对于全局标签与切割位点的处理，输入序列标签主要为了修改
# 对于磷酸化检测来说意义暂时不大
def report_metrics_kingdom_averaged(
        true_global_labels: np.ndarray,
        pred_global_labels: np.ndarray,
        true_sequence_labels: np.ndarray,
        pred_sequence_labels: np.ndarray,
        kingdom_ids: np.ndarray,
        input_token_ids: np.ndarray,
        cleavage_sites: np.ndarray = None,
        use_cs_tag=False,
) -> Dict[str, float]:
    """Utility function to get metrics from model output"""

    sp_tokens = [3, 7, 11, 15, 19]
    if use_cs_tag:
        sp_tokens = [4, 9, 14]
    if (
            cleavage_sites is not None
    ):  # implicit: when cleavage sites are provided, am using region states
        sp_tokens = [5, 11, 19, 26, 31]
        true_cs = cleavage_sites.astype(float)
        # need to convert so np.isnan works
        true_cs[true_cs == -1] = np.nan
    else:
        true_cs = tagged_seq_to_cs_multiclass(true_sequence_labels, sp_tokens=sp_tokens)

    pred_cs = tagged_seq_to_cs_multiclass(pred_sequence_labels, sp_tokens=sp_tokens)

    cs_kingdom = kingdom_ids[~np.isnan(true_cs)]
    pred_cs = pred_cs[~np.isnan(true_cs)]
    true_cs = true_cs[~np.isnan(true_cs)]
    true_cs[np.isnan(true_cs)] = -1
    pred_cs[np.isnan(pred_cs)] = -1

    # applying a threhold of 0.25 (SignalP) to a 4 class case is equivalent to the argmax.
    pred_global_labels_thresholded = pred_global_labels.argmax(axis=1)

    # compute metrics for each kingdom
    rev_kingdom_dict = dict(
        zip(SIGNALP_KINGDOM_DICT.values(), SIGNALP_KINGDOM_DICT.keys())
    )
    all_cs_mcc = []
    all_detection_mcc = []
    metrics_dict = {}
    for kingdom in np.unique(kingdom_ids):
        kingdom_global_labels = true_global_labels[kingdom_ids == kingdom]
        kingdom_pred_global_labels_thresholded = pred_global_labels_thresholded[
            kingdom_ids == kingdom
            ]
        kingdom_true_cs = true_cs[cs_kingdom == kingdom]
        kingdom_pred_cs = pred_cs[cs_kingdom == kingdom]

        metrics_dict[f"CS Recall {rev_kingdom_dict[kingdom]}"] = recall_score(
            kingdom_true_cs, kingdom_pred_cs, average="micro"
        )
        metrics_dict[f"CS Precision {rev_kingdom_dict[kingdom]}"] = precision_score(
            kingdom_true_cs, kingdom_pred_cs, average="micro"
        )
        metrics_dict[f"CS MCC {rev_kingdom_dict[kingdom]}"] = matthews_corrcoef(
            kingdom_true_cs, kingdom_pred_cs
        )
        metrics_dict[f"Detection MCC {rev_kingdom_dict[kingdom]}"] = matthews_corrcoef(
            kingdom_global_labels, kingdom_pred_global_labels_thresholded
        )

        all_cs_mcc.append(metrics_dict[f"CS MCC {rev_kingdom_dict[kingdom]}"])
        all_detection_mcc.append(
            metrics_dict[f"Detection MCC {rev_kingdom_dict[kingdom]}"]
        )

    if (
            cleavage_sites is not None
    ):  # implicit: when cleavage sites are provided, am using region states
        n_h, h_c = class_aware_cosine_similarities(
            pred_sequence_labels,
            input_token_ids,
            true_global_labels,
            replace_value=np.nan,
            op_mode="numpy",
        )
        n_lengths, h_lengths, c_lengths = get_region_lengths(
            pred_sequence_labels, true_global_labels, agg_fn="none"
        )
        for label in np.unique(true_global_labels):
            if label == 0 or label == 5:
                continue

            metrics_dict[f"Cosine similarity nh {label}"] = np.nanmean(
                n_h[true_global_labels == label]
            )
            metrics_dict[f"Cosine similarity hc {label}"] = np.nanmean(
                h_c[true_global_labels == label]
            )
            metrics_dict[f"Average length n {label}"] = n_lengths[
                true_global_labels == label
                ].mean()
            metrics_dict[f"Average length h {label}"] = h_lengths[
                true_global_labels == label
                ].mean()
            metrics_dict[f"Average length c {label}"] = c_lengths[
                true_global_labels == label
                ].mean()
            # w&b can plot histogram heatmaps over time when logging sequences
            metrics_dict[f"Lengths n {label}"] = n_lengths[true_global_labels == label]
            metrics_dict[f"Lengths h {label}"] = h_lengths[true_global_labels == label]
            metrics_dict[f"Lengths c {label}"] = c_lengths[true_global_labels == label]

    metrics_dict["CS MCC"] = sum(all_cs_mcc) / len(all_cs_mcc)
    metrics_dict["Detection MCC"] = sum(all_detection_mcc) / len(all_detection_mcc)

    return metrics_dict


# 针对上面两种，删去全局标签与切割位点的计算，转而研究sequence_label
def report_metrics_pa(
        true_sequence_labels: np.ndarray,
        pred_sequence_labels: np.ndarray,
) -> Dict[str, float]:
    """Utility function to get metrics from model output"""

    # 计算评估指标
    metrics_dict = {}
    metrics_dict["Recall"] = recall_score(true_sequence_labels, pred_sequence_labels, average="micro")
    metrics_dict["Precision"] = precision_score(true_sequence_labels, pred_sequence_labels, average="micro")
    metrics_dict["MCC"] = matthews_corrcoef(true_sequence_labels, pred_sequence_labels)

    return metrics_dict


def train(
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        args: argparse.ArgumentParser,
        global_step: int,
) -> Tuple[float, int]:
    """Predict one minibatch and performs update step.
        该函数预测一个小批量数据并执行更新步骤
    Returns:
        loss: loss value of the minibatch
        返回值是小批量数据的损失值
    输入为：
        model: 要训练的模型。
        train_data: 训练数据的 DataLoader。
        optimizer: 优化器。
        args: 参数解析器，包含训练的各种配置。
        global_step: 全局步数计数器。
    """

    # 设置模型为训练模式
    model.train()
    # 清零优化器的梯度
    optimizer.zero_grad()

    # 初始化存储变量
    all_targets = []  # 存储每个批次的序列标签（sequence labels）
    all_global_targets = []  # 全局标签
    all_global_probs = []  # 全局标签的概率
    all_pos_preds = []  # 存储每个批次的序列标签的预测结果
    all_kingdom_ids = []  # gather ids for kingdom-averaged metrics
    all_token_ids = []  # 存储每个批次的输入数据（token IDs）。即每个批次的 data 张量，表示输入序列的标记ID
    all_cs = []  # 切割位点
    total_loss = 0

    # 遍历训练数据的小批量
    # 从此前定义的打包函数中：data、targets 和 mask 分别是输入 ID、标签 ID 和掩码的张量。
    # sp_region_labels 参数用于指示是否在训练过程中使用信号肽区域标签，即signal peptide region labels，区别在于是否使用切割位点进行训练
    # 不添加即可，这个参数也不会被加入
    # 但是需要删除kingdom_ids与global_targets
    for i, batch in enumerate(train_data):
        if args.sp_region_labels:
            (
                data,
                targets,
                input_mask,
                global_targets,
                cleavage_sites,
                sample_weights,
                kingdom_ids,
            ) = batch
        else:
            (
                data,
                targets,
                input_mask,
                global_targets,
                sample_weights,
                kingdom_ids,
            ) = batch

        # 将数据和标签传递到指定的设备上
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        global_targets = global_targets.to(device)
        sample_weights = sample_weights.to(device) if args.use_sample_weights else None
        kingdom_ids = kingdom_ids.to(device)

        # 清零优化器的梯度。
        # 前向传播，计算损失和预测结果。
        # 如果使用 DataParallel，损失是一个向量，需要取平均值，累加总损失。
        optimizer.zero_grad()

        # 向前传播的过程，这里需要删去global_probs,因为不得出这一结果
        loss, global_probs, pos_probs, pos_preds = model(
            data,
            global_targets=None,
            targets=targets if not args.sp_region_labels else None,
            targets_bitmap=targets if args.sp_region_labels else None,
            input_mask=input_mask,
            sample_weights=sample_weights,
            kingdom_ids=kingdom_ids if args.kingdom_embed_size > 0 else None,
        )
        loss = (
            loss.mean()
        )  # if DataParallel because loss is a vector, if not doesn't matter
        # 同样的，删去global
        total_loss += loss.item()
        all_targets.append(targets.detach().cpu().numpy())
        all_global_targets.append(global_targets.detach().cpu().numpy())
        all_global_probs.append(global_probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_kingdom_ids.append(kingdom_ids.detach().cpu().numpy())
        all_token_ids.append(data.detach().cpu().numpy())
        all_cs.append(cleavage_sites if args.sp_region_labels else None)

        # 如果 args.sp_region_labels 为 True 且 args.region_regularization_alpha 大于 0，则进行区域正则化。
        # if args.region_regularization_alpha >0:
        # removing special tokens by indexing should be sufficient.
        # remaining SEP tokens (when sequence was padded) are ignored in aggregation.
        if args.sp_region_labels and args.region_regularization_alpha > 0:
            nh, hc = compute_cosine_region_regularization(
                pos_probs, data[:, 2:-1], global_targets, input_mask[:, 2:-1]
            )
            loss = loss + nh.mean() * args.region_regularization_alpha
            loss = loss + hc.mean() * args.region_regularization_alpha
            log_metrics(
                {
                    "n-h regularization": nh.mean().detach().cpu().numpy(),
                    "h-c regularization": hc.mean().detach().cpu().numpy(),
                },
                "train",
                global_step,
            )

        # 反向传播
        loss.backward()

        # 梯度裁剪、步骤优化与信息记录
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        # from IPython import embed; embed()
        optimizer.step()

        log_metrics({"loss": loss.item()}, "train", global_step)

        if args.optimizer == "smart_adamax":
            log_metrics({"Learning rate": optimizer.get_lr()[0]}, "train", global_step)
        else:
            log_metrics(
                {"Learning Rate": optimizer.param_groups[0]["lr"]}, "train", global_step
            )
        global_step += 1

    # 删除部分
    all_targets = np.concatenate(all_targets)
    all_global_targets = np.concatenate(all_global_targets)
    all_global_probs = np.concatenate(all_global_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_kingdom_ids = np.concatenate(all_kingdom_ids)
    all_token_ids = np.concatenate(all_token_ids)
    all_cs = np.concatenate(all_cs) if args.sp_region_labels else None

    # if args.average_per_kingdom:
    #     metrics = report_metrics_kingdom_averaged(
    #         all_global_targets,
    #         all_global_probs,
    #         all_targets,
    #         all_pos_preds,
    #         all_kingdom_ids,
    #         all_token_ids,
    #         all_cs,
    #         args.use_cs_tag,
    #     )
    # else:
    #     metrics = report_metrics(
    #         all_global_targets,
    #         all_global_probs,
    #         all_targets,
    #         all_pos_preds,
    #         args.use_cs_tag,
    #     )

    # 尝试，只保留了序列处理的部分
    metrics = report_metrics_pa(
        all_targets,
        all_pos_preds,
    )

    log_metrics(metrics, "train", global_step)

    return total_loss / len(train_data), global_step
