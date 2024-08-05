import argparse
import os
import time
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

script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))


# 下面是自定义的库的引入
from pa_detect import (
    LargeCRFPartitionDataset,
    RegionCRFDataset,
    SIGNALP6_GLOBAL_LABEL_DICT,
    SIGNALP_KINGDOM_DICT,
    PhosphoicAcidThreeLineFastaDataset,
    compute_cosine_region_regularization,
    RegionCRFDatasetPA,
    LargeCRFPartitionDatasetPA,
    Adamax,
    ProteinBertTokenizer,
    BertSequenceTaggingCRF,
    class_aware_cosine_similarities,
    get_region_lengths
)



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


# 针对上面两种，删去全局标签与切割位点的计算，转而研究sequence_label，目前没有投入使用
# Todo 需要将形状由[batchsize, length, cls]改为[length, cls]然后再去求recall
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
    # 原本的为：data,
    #                 targets,
    #                 input_mask,
    #                 global_targets,
    #                 sample_weights,
    #                 kingdom_ids,
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
                sample_weights,
            ) = batch

        # 将数据和标签传递到指定的设备上
        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        # global_targets = global_targets.to(device)
        sample_weights = sample_weights.to(device) if args.use_sample_weights else None
        # kingdom_ids = kingdom_ids.to(device)

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
    # all_targets = np.concatenate(all_targets)
    # all_global_targets = np.concatenate(all_global_targets)
    # all_global_probs = np.concatenate(all_global_probs)
    # all_pos_preds = np.concatenate(all_pos_preds)
    # all_kingdom_ids = np.concatenate(all_kingdom_ids)
    # all_token_ids = np.concatenate(all_token_ids)
    # all_cs = np.concatenate(all_cs) if args.sp_region_labels else None

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
    # 暂不使用，需要修改np数组的格式，后面尝试加上
    # metrics = report_metrics_pa(
    #     all_targets,
    #     all_pos_preds,
    # )

    # log_metrics(metrics, "train", global_step)

    return total_loss / len(train_data), global_step


def validate(model: torch.nn.Module, valid_data: DataLoader, args) -> float:
    """Run over the validation data. Average loss over the full set."""
    # 用于在验证数据集上运行模型，并计算整个数据集的平均损失
    # 将模型设置为评估模式，关闭dropout等
    model.eval()

    all_targets = []
    all_probs = []
    all_pos_preds = []
    all_token_ids = []

    total_loss = 0
    for i, batch in enumerate(valid_data):
        data, targets, input_mask, sample_weights = batch

        data = data.to(device)
        targets = targets.to(device)
        input_mask = input_mask.to(device)
        sample_weights = sample_weights.to(device) if args.use_sample_weights else None

        with torch.no_grad():
            loss, probs, pos_probs, pos_preds = model(
                data,
                targets=targets,
                sample_weights=sample_weights,
                input_mask=input_mask,
            )

        total_loss += loss.mean().item()
        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        all_pos_preds.append(pos_preds.detach().cpu().numpy())
        all_token_ids.append(data.detach().cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    all_pos_preds = np.concatenate(all_pos_preds)
    all_token_ids = np.concatenate(all_token_ids)

    # metrics = report_metrics(
    #     all_targets,
    #     all_probs,
    #     all_pos_preds,
    #     args.use_cs_tag,
    # )

    # val_metrics = {"loss": total_loss / len(valid_data), **metrics}
    # 只保留平均损失
    return (total_loss / len(valid_data))


def main_training_loop(args: argparse.ArgumentParser):

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    logger = setup_logger()
    f_handler = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
    )
    f_handler.setFormatter(formatter)

    logger.addHandler(f_handler)

    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    experiment_name = f"{args.experiment_name}_{args.test_partition}_{args.validation_partition}_{time_stamp}"

    # TODO get rid of this dirty fix once wandb works again
    global wandb
    import wandb

    if (
        wandb.run is None and not args.crossval_run
    ):  # Only initialize when there is no run yet (when importing main_training_loop to other scripts)
        wandb.init(dir=args.output_dir, name=experiment_name)
    else:
        wandb = DecoyWandb()

    # Set seed
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        seed = args.random_seed
    else:
        seed = torch.seed()

    logger.info(f"torch seed: {seed}")
    wandb.config.update({"seed": seed})

    logger.info(f"Saving to {args.output_dir}")

    # Setup Model
    logger.info(f"Loading pretrained model in {args.resume}")
    config = MODEL_DICT[args.model_architecture][0].from_pretrained(args.resume)

    # if config.xla_device:
    #     setattr(config, "xla_device", False)
    if not hasattr(config, "xla_device"):
        setattr(config,"xla_device",False)

    setattr(config, "num_labels", args.num_seq_labels)
    setattr(config, "num_global_labels", args.num_global_labels)

    setattr(config, "lm_output_dropout", args.lm_output_dropout)
    setattr(config, "lm_output_position_dropout", args.lm_output_position_dropout)
    setattr(config, "crf_scaling_factor", args.crf_scaling_factor)
    setattr(
        config, "use_large_crf", True
    )  # legacy, parameter is used in evaluation scripts. Ensures choice of right CS states.

    if args.sp_region_labels:
        setattr(config, "use_region_labels", True)

    # 这个默认被初始化为0
    if args.kingdom_embed_size > 0:
        setattr(config, "use_kingdom_id", True)
        setattr(config, "kingdom_embed_size", args.kingdom_embed_size)

    # hardcoded for full model, 5 classes, 37 tags
    # 这个标签的作用是Use labels for n,h,c regions of SPs，即要去区分nhc区用的
    # 如果指定了就是true，不指定即可，为了以防万一我也准备好了针对磷酸化的硬编码
    if args.constrain_crf and args.sp_region_labels:
        allowed_transitions = [
            # NO_SP
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 2),
            (1, 0),
            (2, 1),
            (2, 2),  # I-I, I-M, M-M, M-O, M-I, O-M, O-O
            # SPI
            # 3 N, 4 H, 5 C, 6 I, 7M, 8 O
            (3, 3),
            (3, 4),
            (4, 4),
            (4, 5),
            (5, 5),
            (5, 8),
            (8, 8),
            (8, 7),
            (7, 7),
            (7, 6),
            (6, 6),
            (6, 7),
            (7, 8),
            # SPII
            # 9 N, 10 H, 11 CS, 12 C1, 13 I, 14 M, 15 O
            (9, 9),
            (9, 10),
            (10, 10),
            (10, 11),
            (11, 11),
            (11, 12),
            (12, 15),
            (15, 15),
            (15, 14),
            (14, 14),
            (14, 13),
            (13, 13),
            (13, 14),
            (14, 15),
            # TAT
            # 16 N, 17 RR, 18 H, 19 C, 20 I, 21 M, 22 O
            (16, 16),
            (16, 17),
            (17, 17),
            (17, 16),
            (16, 18),
            (18, 18),
            (18, 19),
            (19, 19),
            (19, 22),
            (22, 22),
            (22, 21),
            (21, 21),
            (21, 20),
            (20, 20),
            (20, 21),
            (21, 22),
            # TATLIPO
            # 23 N, 24 RR, 25 H, 26 CS, 27 C1, 28 I, 29 M, 30 O
            (23, 23),
            (23, 24),
            (24, 24),
            (24, 23),
            (23, 25),
            (25, 25),
            (25, 26),
            (26, 26),
            (26, 27),
            (27, 30),
            (30, 30),
            (30, 29),
            (29, 29),
            (29, 28),
            (28, 28),
            (28, 29),
            (29, 30),
            # PILIN
            # 31 P, 32 CS, 33 H, 34 I, 35 M, 36 O
            (31, 31),
            (31, 32),
            (32, 32),
            (32, 33),
            (33, 33),
            (33, 36),
            (36, 36),
            (36, 35),
            (35, 35),
            (35, 34),
            (34, 34),
            (34, 35),
            (35, 36),
        ]
        # 这个部分在预训练数据进行了标注：>Q8TF40|EUKARYA|NO_SP|0,磷酸化检测不需要这样划分
        #            'NO_SP_I' : 0,
        #            'NO_SP_M' : 1,
        #            'NO_SP_O' : 2,
        allowed_starts = [0, 2, 3, 9, 16, 23, 31]
        allowed_ends = [0, 1, 2, 13, 14, 15, 20, 21, 22, 28, 29, 30, 34, 35, 36]

        # 这里暂时设置四种情况
        # 我需要硬编码吗？需要，这个就相当于训练集里面可以0-1也可以1-0，那自然是可以的
        # allowed_transitions = [
        #     (0, 0),  # 非磷酸化可以保持非磷酸化
        #     (1, 1),  # 磷酸化可以保持磷酸化
        #     (0, 1),  # 非磷酸化可以转变为磷酸化
        #     (1, 0),  # 磷酸化可以转变为非磷酸化
        # ]

        # allowed_starts = [0, 1]
        # allowed_ends = [0, 1]

        setattr(config, "allowed_crf_transitions", allowed_transitions)
        setattr(config, "allowed_crf_starts", allowed_starts)
        setattr(config, "allowed_crf_ends", allowed_ends)

    # setattr(config, 'gradient_checkpointing', True) #hardcoded when working with 256aa data
    # 这两个都不指定，也不用删除
    if args.kingdom_as_token:
        setattr(
            config, "kingdom_id_as_token", True
        )  # model needs to know that token at pos 1 needs to be removed for CRF

    if args.global_label_as_input:
        setattr(config, "type_id_as_token", True)

    if args.remove_top_layers > 0:
        # num_hidden_layers if bert
        n_layers = (
            config.num_hidden_layers
            if args.model_architecture == "bert_prottrans"
            else config.n_layer
        )
        if args.remove_top_layers > n_layers:
            logger.warning(f"Trying to remove more layers than there are: {n_layers}")
            args.remove_top_layers = n_layers

        setattr(
            config,
            "num_hidden_layers"
            if args.model_architecture == "bert_prottrans"
            else "n_layer",
            n_layers - args.remove_top_layers,
        )

    model = MODEL_DICT[args.model_architecture][1].from_pretrained(
        args.resume, config=config
    )
    tokenizer = TOKENIZER_DICT[args.model_architecture][0].from_pretrained(
        TOKENIZER_DICT[args.model_architecture][1], do_lower_case=False
    )
    logger.info(
        f"Loaded weights from {args.resume} for model {model.base_model_prefix}"
    )

    if args.kingdom_as_token:
        logger.info(
            "Using kingdom IDs as word in sequence, extending embedding layer of pretrained model."
        )
        tokenizer = TOKENIZER_DICT[args.model_architecture][0].from_pretrained(
            "data/tokenizer", do_lower_case=False
        )
        model.resize_token_embeddings(tokenizer.tokenizer.vocab_size)

    # setup data
    val_id = args.validation_partition
    test_id = args.test_partition
    train_ids = [0, 1, 2]  # ,3,4]
    train_ids.remove(val_id)
    train_ids.remove(test_id)
    logger.info(f"Training on {train_ids}, validating on {val_id}")

    kingdoms = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"]

    # 添加这个标签后，就会转而使用分区CRF
    if args.sp_region_labels:
        train_data = RegionCRFDataset(
            args.data,
            args.sample_weights,
            tokenizer=tokenizer,
            partition_id=train_ids,
            kingdom_id=kingdoms,
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=args.positive_samples_weight,
            make_cs_state=args.use_cs_tag,
            add_global_label=args.global_label_as_input,
            augment_data_paths=[args.additional_train_set],
        )
        val_data = RegionCRFDataset(
            args.data,
            args.sample_weights,
            tokenizer=tokenizer,
            partition_id=[val_id],
            kingdom_id=kingdoms,
            add_special_tokens=True,
            return_kingdom_ids=True,
            positive_samples_weight=args.positive_samples_weight,
            make_cs_state=args.use_cs_tag,
            add_global_label=args.global_label_as_input,
        )
        logger.info("Using labels for SP region prediction.")
    else:
        # data_path: Union[str, Path],
        # sample_weights_path = None,
        # tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
        # partition_id: List[str] = [0, 1, 2],
        # add_special_tokens = False,
        # one_versus_all = False,
        # positive_samples_weight = None,
        # return_kingdom_ids = False,
        # make_cs_state = False,

        # 添加了参数中的标签就是True，不要添加，这是求切割位点的
        train_data = LargeCRFPartitionDatasetPA(
            args.data,
            args.sample_weights,
            tokenizer=tokenizer,
            partition_id=train_ids,
            add_special_tokens=False,
            return_kingdom_ids=False,
            positive_samples_weight=args.positive_samples_weight,
            make_cs_state=args.use_cs_tag,
        )
        val_data = LargeCRFPartitionDataset(
            args.data,
            args.sample_weights,
            tokenizer=tokenizer,
            partition_id=[val_id],
            add_special_tokens=False,
            return_kingdom_ids=False,
            positive_samples_weight=args.positive_samples_weight,
            make_cs_state=args.use_cs_tag,
        )
    logger.info(
        f"{len(train_data)} training sequences, {len(val_data)} validation sequences."
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=train_data.collate_fn,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, collate_fn=train_data.collate_fn
    )

    # use sample weights to load random samples as minibatches according to weights,不需要添加
    if args.use_random_weighted_sampling:
        sampler = WeightedRandomSampler(
            train_data.sample_weights, len(train_data), replacement=False
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            collate_fn=train_data.collate_fn,
            sampler=sampler,
        )
    elif args.use_weighted_kingdom_sampling:
        sampler = WeightedRandomSampler(
            train_data.balanced_sampling_weights, len(train_data), replacement=False
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            collate_fn=train_data.collate_fn,
            sampler=sampler,
        )
        logger.info(
            f"Using kingdom-balanced oversampling. Sum of all sampling weights = {sum(train_data.balanced_sampling_weights)}"
        )

    logger.info(f"Data loaded. One epoch = {len(train_loader)} batches.")

    # set up wandb logging, login and project id from commandline vars
    wandb.config.update(args)
    wandb.config.update({"git commit ID": GIT_HASH})
    wandb.config.update(model.config.to_dict())
    # TODO uncomment as soon as w&b fixes the bug on their end.
    # wandb.watch(model)
    logger.info(f"Logging experiment as {experiment_name} to wandb/tensorboard")
    logger.info(f"Saving checkpoints at {args.output_dir}")

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    if args.optimizer == "adamax":
        optimizer = torch.optim.Adamax(
            model.parameters(), lr=args.lr, weight_decay=args.wdecay
        )
    # 更换bert版本的Adamax
    if args.optimizer == "smart_adamax":
        t_total = len(train_loader) * args.epochs
        optimizer = Adamax(
            model.parameters(),
            lr=args.lr,
            warmup=0.1,
            t_total=t_total,
            schedule="warmup_linear",
            betas=(0.9, 0.999),
            weight_decay=args.wdecay,
            max_grad_norm=1,
        )

    model.to(device)
    logger.info("Model set up!")
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_parameters} trainable parameters")

    logger.info(f"Running model on {device}, not using nvidia apex")

    # keep track of best loss
    stored_loss = 100000000
    learning_rate_steps = 0
    num_epochs_no_improvement = 0
    global_step = 0
    best_mcc_sum = 0
    best_mcc_global = 0
    best_mcc_cs = 0
    for epoch in range(1, args.epochs + 1):
        logger.info(f"Starting epoch {epoch}")

        epoch_loss, global_step = train(
            model, train_loader, optimizer, args, global_step
        )

        logger.info(
            f"Step {global_step}, Epoch {epoch}: validating for {len(val_loader)} Validation steps"
        )
        val_loss, val_metrics = validate(model, val_loader, args)
        log_metrics(val_metrics, "val", global_step)
        logger.info(
            f"Validation: MCC global {val_metrics['Detection MCC']}, MCC seq {val_metrics['CS MCC']}. Epochs without improvement: {num_epochs_no_improvement}. lr step {learning_rate_steps}"
        )

        mcc_sum = val_metrics["Detection MCC"] + val_metrics["CS MCC"]
        log_metrics({"MCC Sum": mcc_sum}, "val", global_step)
        if mcc_sum > best_mcc_sum:
            best_mcc_sum = mcc_sum
            best_mcc_global = val_metrics["Detection MCC"]
            best_mcc_cs = val_metrics["CS MCC"]
            num_epochs_no_improvement = 0

            model.save_pretrained(args.output_dir)
            logger.info(
                f'New best model with loss {val_loss},MCC Sum {mcc_sum} MCC global {val_metrics["Detection MCC"]}, MCC seq {val_metrics["CS MCC"]}, Saving model, training step {global_step}'
            )

        else:
            num_epochs_no_improvement += 1

        # when cross-validating, check that the seed is working for region detection
        if args.crossval_run and epoch == 1:
            # small length in first epoch = bad seed.
            if val_metrics["Average length n 1"] <= 4:
                print("Bad seed for region tagging.")
                run_completed = False
                return best_mcc_global, best_mcc_cs, run_completed

    logger.info(f"Epoch {epoch}, epoch limit reached. Training complete")
    logger.info(
        f"Best: MCC Sum {best_mcc_sum}, Detection {best_mcc_global}, CS {best_mcc_cs}"
    )
    log_metrics(
        {
            "Best MCC Detection": best_mcc_global,
            "Best MCC CS": best_mcc_cs,
            "Best MCC sum": best_mcc_sum,
        },
        "val",
        global_step,
    )

    # print_all_final_metrics = True
    # if print_all_final_metrics == True:
    #     # reload best checkpoint
    #     model = MODEL_DICT[args.model_architecture][1].from_pretrained(args.output_dir)
    #     ds = RegionCRFDataset(
    #         args.data,
    #         args.sample_weights,
    #         tokenizer=tokenizer,
    #         partition_id=[test_id],
    #         kingdom_id=kingdoms,
    #         add_special_tokens=True,
    #         return_kingdom_ids=True,
    #         positive_samples_weight=args.positive_samples_weight,
    #         make_cs_state=args.use_cs_tag,
    #         add_global_label=args.global_label_as_input,
    #     )
    #     dataloader = torch.utils.data.DataLoader(
    #         ds, collate_fn=ds.collate_fn, batch_size=80
    #     )
    #     metrics = get_metrics_multistate(model, dataloader)
    #     val_metrics = get_metrics_multistate(model, val_loader)
    #
    #     if args.crossval_run or args.log_all_final_metrics:
    #         log_metrics(metrics, "test", global_step)
    #         log_metrics(val_metrics, "best_val", global_step)
    #     logger.info(metrics)
    #     logger.info("Validation set")
    #     logger.info(val_metrics)

        ## prettyprint everythingh
        # import pandas as pd
        #
        # # df = pd.DataFrame.from_dict(x, orient='index')
        # # df.index = df.index.str.split('_', expand=True)
        # # print(df.sort_index())
        #
        # df = pd.DataFrame.from_dict([metrics, val_metrics]).T
        # df.columns = ["test", "val"]
        # df.index = df.index.str.split("_", expand=True)
        # pd.set_option("display.max_rows", None)
        # print(df.sort_index())

    run_completed = True
    return best_mcc_global, best_mcc_cs, run_completed  # best_mcc_sum


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train Bert-CRF model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/data/train_set.fasta",
        help="location of the data corpus. Expects test, train and valid .fasta",
    )
    parser.add_argument(
        "--sample_weights",
        type=str,
        default=None,
        help="path to .csv file with the weights for each sample",
    )
    parser.add_argument(
        "--test_partition",
        type=int,
        default=0,
        help="partition that will not be used in this training run",
    )
    parser.add_argument(
        "--validation_partition",
        type=int,
        default=1,
        help="partition that will be used for validation in this training run",
    )

    # args relating to training strategy.
    parser.add_argument("--lr", type=float, default=10, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=8000, help="upper epoch limit")

    # batch_size默认为80
    parser.add_argument(
        "--batch_size", type=int, default=80, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--wdecay",
        type=float,
        default=1.2e-6,
        help="weight decay applied to all weights",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        help="optimizer to use (sgd, adam, adamax)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="training_run",
        help="path to save logs and trained model",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="Rostlab/prot_bert",
        help="path of model to resume (directory containing .bin and config.json, or HF model)",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="BERT-CRF",
        help="experiment name for logging",
    )
    parser.add_argument(
        "--crossval_run",
        action="store_true",
        help="override name with timestamp, save with split identifiers. Use when making checkpoints for crossvalidation.",
    )
    parser.add_argument(
        "--log_all_final_metrics",
        action="store_true",
        help="log all final test/val metrics to w&b",
    )

    parser.add_argument("--num_seq_labels", type=int, default=37)
    parser.add_argument("--num_global_labels", type=int, default=6)
    parser.add_argument(
        "--global_label_as_input",
        action="store_true",
        help="Add the global label to the input sequence (only predict CS given a known label)",
    )

    parser.add_argument(
        "--region_regularization_alpha",
        type=float,
        default=0,
        help="multiplication factor for the region similarity regularization term",
    )
    parser.add_argument(
        "--lm_output_dropout",
        type=float,
        default=0.1,
        help="dropout applied to LM output",
    )
    parser.add_argument(
        "--lm_output_position_dropout",
        type=float,
        default=0.1,
        help="dropout applied to LM output, drops full hidden states from sequence",
    )
    parser.add_argument(
        "--use_sample_weights",
        action="store_true",
        help="Use sample weights to rescale loss per sample",
    )
    parser.add_argument(
        "--use_random_weighted_sampling",
        action="store_true",
        help="use sample weights to load random samples as minibatches according to weights",
    )
    parser.add_argument(
        "--positive_samples_weight",
        type=float,
        default=None,
        help="Scaling factor for positive samples loss, e.g. 1.5. Needs --use_sample_weights flag in addition.",
    )
    parser.add_argument(
        "--average_per_kingdom",
        action="store_true",
        help="Average MCCs per kingdom instead of overall computatition",
    )
    parser.add_argument(
        "--crf_scaling_factor",
        type=float,
        default=1.0,
        help="Scale CRF NLL by this before adding to global label loss",
    )
    parser.add_argument(
        "--use_weighted_kingdom_sampling",
        action="store_true",
        help="upsample all kingdoms to equal probabilities",
    )
    parser.add_argument(
        "--random_seed", type=int, default=None, help="random seed for torch."
    )
    parser.add_argument(
        "--additional_train_set",
        type=str,
        default=None,
        help="Additional samples to train on",
    )

    # args for model architecture
    parser.add_argument(
        "--model_architecture",
        type=str,
        default="bert_prottrans",
        help="which model architecture the checkpoint is for",
    )
    parser.add_argument(
        "--remove_top_layers",
        type=int,
        default=0,
        help="How many layers to remove from the top of the LM.",
    )
    parser.add_argument(
        "--kingdom_embed_size",
        type=int,
        default=0,
        help="If >0, embed kingdom ids to N and concatenate with LM hidden states before CRF.",
    )
    parser.add_argument(
        "--use_cs_tag",
        action="store_true",
        help="Replace last token of SP with C for cleavage site",
    )
    parser.add_argument(
        "--kingdom_as_token",
        action="store_true",
        help="Kingdom ID is first token in the sequence",
    )
    parser.add_argument(
        "--sp_region_labels",
        action="store_true",
        help="Use labels for n,h,c regions of SPs.",
    )
    parser.add_argument(
        "--constrain_crf",
        action="store_true",
        help="Constrain the transitions of the region-tagging CRF.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # make unique output dir in output dir
    time_stamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())
    full_name = "_".join(
        [
            args.experiment_name,
            "test",
            str(args.test_partition),
            "valid",
            str(args.validation_partition),
            time_stamp,
        ]
    )

    if args.crossval_run == True:
        full_name = "_".join(
            [
                args.experiment_name,
                "test",
                str(args.test_partition),
                "valid",
                str(args.validation_partition),
            ]
        )

    args.output_dir = os.path.join(args.output_dir, full_name)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    main_training_loop(args)