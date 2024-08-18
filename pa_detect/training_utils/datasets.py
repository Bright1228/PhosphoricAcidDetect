# © Copyright Technical University of Denmark
"""
Dataset to deal with the 3-line fasta format used in SignalP.
"""
import torch
from torch.utils.data import Dataset
from typing import Union, List, Dict, Any, Sequence
import numpy as np
import pandas as pd
from pathlib import Path
from transformers import PreTrainedTokenizer
from collections import defaultdict

from .label_processing_utils import process_SP
from .label_processing_utils import process_PA

# 这里是定义信号肽类型的，用于预测，在此定义一个新的结构用于判断磷酸化
# [S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]
SIGNALP_VOCAB = [
    "S",
    "I",
    "M",
    "O",
    "T",
    "L",
]  # NOTE eukarya only uses {'I', 'M', 'O', 'S'}
SIGNALP_GLOBAL_LABEL_DICT = {"NO_SP": 0, "SP": 1, "LIPO": 2, "TAT": 3}
SIGNALP_KINGDOM_DICT = {"EUKARYA": 0, "POSITIVE": 1, "NEGATIVE": 2, "ARCHAEA": 3}
SIGNALP6_GLOBAL_LABEL_DICT = {
    "NO_SP": 0,
    "SP": 1,
    "LIPO": 2,
    "TAT": 3,
    "TATLIPO": 4,
    "PILIN": 5,
}

# 此处定义磷酸化判断的相关结构
# 磷酸化词汇表，这里的01会出现在fasta文件的第三行
PHOSPHO_VOCAB = [
    "0",  # 未磷酸化
    "1",  # 磷酸化
]
# 提供全局标签与描述的映射提供全局标签与描述的映射的字典
PHOSPHO_GLOBAL_LABEL_DICT = {
    "UNPHOSPHORYLATED": 0,  # 未磷酸化的全局标签
    "PHOSPHORYLATED": 1,    # 磷酸化的全局标签
}
def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    print("执行了函数pad sequences!")
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

# 处理示例
"""
(
    ['>Q99PF5|EUKARYA|NO_SP|0'],
    ['MSDYSTGGPPPGPPPPAGGGGGAAGAGGGPPPGPPGAGDRGGGGPGGGGPGGGGASGGPSQPPGGGGPGI'],
    ['IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII']
)

"""
def parse_threeline_fasta(filepath: Union[str, Path]):

    with open(filepath, "r") as f:
        # 读取文件并删除换行符
        lines = f.read().splitlines()  # f.readlines()
        # 第一行是标识符
        identifiers = lines[::3]
        # 第二行是蛋白质序列
        sequences = lines[1::3]
        # 第三行是标签
        labels = lines[2::3]

    return identifiers, sequences, labels

# 这个类会对提供的具体的数据进行操作
# >A8FLJ3|NEGATIVE|TAT|0   将会被拆解为：
# acc_ids = A8FLJ3
# kingdom_ids = NEGATIVE
# type_ids = TAT
# partition_ids = 0
def subset_dataset(
    identifiers: List[str],
    sequences: List[str],
    labels: List[str],
    partition_id: List[int],
    kingdom_id: List[str],
    type_id: List[str],
):
    """Extract a subset from the complete .fasta dataset"""

    # break up the identifier into elements
    parsed = [element.lstrip(">").split("|") for element in identifiers]
    acc_ids, kingdom_ids, type_ids, partition_ids = [
        np.array(x) for x in list(zip(*parsed))
    ]
    partition_ids = partition_ids.astype(int)

    king_idx = np.isin(kingdom_ids, kingdom_id)
    part_idx = np.isin(partition_ids, partition_id)
    type_idx = np.isin(type_ids, type_id)

    select_idx = king_idx & part_idx & type_idx
    assert select_idx.sum() > 0, "This id combination does not yield any sequences!"

    # index in numpy, and then return again as lists.
    identifiers_out, sequences_out, labels_out = [
        list(np.array(x)[select_idx]) for x in [identifiers, sequences, labels]
    ]

    return identifiers_out, sequences_out, labels_out

# 实际上这个函数涉及到区分训练集，还是要用的
# 实际的训练集中，identifiers表示为protein_id|...|partition_id
def subset_dataset_PA(
    identifiers: List[str],
    sequences: List[str],
    labels: List[str],
    partition_id: List[int],
):
    """Extract a subset from the complete .fasta dataset based on partition_id"""

    # break up the identifier into elements
    parsed = [element.lstrip(">").split("|") for element in identifiers]
    acc_ids = [elements[0] for elements in parsed]
    partition_ids = [int(elements[-1]) for elements in parsed]

    # partition_id在调用这个函数时会传入数组，就是所有的集合
    partition_ids = np.array(partition_ids)

    part_idx = np.isin(partition_ids, partition_id)

    select_idx = part_idx
    assert select_idx.sum() > 0, "This partition id combination does not yield any sequences!"

    # index in numpy, and then return again as lists.
    # 这里就是根据筛选结果，保留一部分序列
    identifiers_out, sequences_out, labels_out = [
        list(np.array(x)[select_idx]) for x in [identifiers, sequences, labels]
    ]

    return identifiers_out, sequences_out, labels_out

# 这里进行解码，理论上我换一个字母表就能用了
class SP_label_tokenizer:
    """[S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]"""

    def __init__(self, labels: List[str] = SIGNALP_VOCAB):
        # build mapping
        token_ids = list(range(len(labels)))
        self.vocab = dict(zip(labels, token_ids))

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def sequence_to_token_ids(self, sequence) -> List[int]:
        tokens = self.tokenize(sequence)
        ids = self.convert_tokens_to_ids(tokens)
        return ids


# 自定义磷酸化检测用的解码器
class PhosphoicAcid_label_tokenizer:
    """[1: 发生了磷酸化 | 0: 没有发生磷酸化]"""

    # 修改定义的标签列表，labels默认为磷酸化的字母表
    def __init__(self, labels: List[str] = PHOSPHO_VOCAB):
        # build mapping
        # token_ids与字母表一一对应，即0-0,1-1
        token_ids = list(range(len(labels)))
        # 将上面的token_ids与标签建立映射
        self.vocab = dict(zip(labels, token_ids))

    # 将输入字符串 text 分割成单个字符列表
    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

    # 将单个标记 token 转换为对应的整数 ID
    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    # 将tokens转化为列表
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    # 输入一个字符串
    def sequence_to_token_ids(self, sequence) -> List[int]:
        tokens = self.tokenize(sequence)
        ids = self.convert_tokens_to_ids(tokens)
        return ids

class AbstractThreeLineFastaDataset(Dataset):
    """Abstract Dataset to load a three-line fasta file.
    Need to implement __getitem__ in child classes, to preprocess sequences as needed."""

    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
    ):

        super().__init__()

        if isinstance(tokenizer, str):
            from tape import TAPETokenizer

            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        self.label_tokenizer = SP_label_tokenizer()

        self.data_file = Path(data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        _, self.sequences, self.labels = parse_threeline_fasta(self.data_file)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index):
        raise NotImplementedError

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


# 处理分割位点的类，或许我直接不需要它
# Process this:
# >P10152|EUKARYA|SP|1
# MVMVLSPLLLVFILGLGLTPVAPAQDDYRYIHFLTQHYDAKPKGRNDEYCFNMMKNRRLTRPCKDRNTFI
# SSSSSSSSSSSSSSSSSSSSSSSOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO
# To: 23 (CS = label)
class PointerSentinelThreeLineFastaDataset(AbstractThreeLineFastaDataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
    ):
        super().__init__(data_path, tokenizer)

    def _process_labels(self, labels: List[int]) -> int:
        """Takes a SignalP label sequence and creates a CS position label"""
        sp_pos = np.where(np.array(labels) == 0)[0]  # 0 is token for S
        if len(sp_pos) == 0:
            cs_pos = len(labels)  # target is sentinel
        else:
            cs_pos = sp_pos.max()  # target is CS

        return cs_pos

    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]
        token_ids = self.tokenizer.tokenize(item) + [self.tokenizer.stop_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

        label_ids = self.label_tokenizer.sequence_to_token_ids(labels)
        label = self._process_labels(label_ids)
        return np.array(token_ids), np.array(label)

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(zip(*batch))
        data = torch.from_numpy(pad_sequences(input_ids, 0))

        targets = torch.tensor(np.stack(labels))

        return data, targets


class ThreeLineFastaDataset(Dataset):
    """Creates a dataset from a SignalP format 3-line .fasta file."""
    # "IUPAC" 是指用于编码核酸序列（例如 DNA 和 RNA）的标准字母表。
    # 它包含了标准的核苷酸（A、C、G、T/U）以及表示不确定或变异位置的其他字母（如 R 表示 A 或 G，Y 表示 C 或 T 等）。
    # 不必修改
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
        add_special_tokens=False,
    ):

        super().__init__()
        self.add_special_tokens = add_special_tokens

        if isinstance(tokenizer, str):
            from tape import TAPETokenizer

            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        self.label_tokenizer = SP_label_tokenizer()

        self.data_file = Path(data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        self.identifiers, self.sequences, self.labels = parse_threeline_fasta(
            self.data_file
        )
        self.global_labels = [x.split("|")[2] for x in self.identifiers]


    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index):
        # 获取索引为index的序列
        item = self.sequences[index]
        labels = self.labels[index]
        global_label = self.global_labels[index]
        sample_weight = (
            self.sample_weights[index] if hasattr(self, "sample_weights") else None
        )
        kingdom_id = (
            SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]]
            if hasattr(self, "kingdom_ids")
            else None
        )

        if self.add_special_tokens == True:
            token_ids = self.tokenizer.encode(item)
        else:
            # 只进行标记化处理，然后将标记转换为ID
            token_ids = self.tokenizer.tokenize(item)  # + [self.tokenizer.stop_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

        label_ids = self.label_tokenizer.sequence_to_token_ids(labels)
        global_label_id = SIGNALP_GLOBAL_LABEL_DICT[global_label]

        input_mask = np.ones_like(token_ids)

        return_tuple = (
            np.array(token_ids),
            np.array(label_ids),
            np.array(input_mask),
            global_label_id,
        )

        if sample_weight is not None:
            return_tuple = return_tuple + (sample_weight,)
        if kingdom_id is not None:
            return_tuple = return_tuple + (kingdom_id,)

        return return_tuple

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        # unpack the list of tuples
        if hasattr(self, "sample_weights") and hasattr(self, "kingdom_ids"):
            (
                input_ids,
                label_ids,
                mask,
                global_label_ids,
                sample_weights,
                kingdom_ids,
            ) = tuple(zip(*batch))
        elif hasattr(self, "sample_weights"):
            input_ids, label_ids, mask, global_label_ids, sample_weights = tuple(
                zip(*batch)
            )
        elif hasattr(self, "kingdom_ids"):
            input_ids, label_ids, mask, global_label_ids, kingdom_ids = tuple(
                zip(*batch)
            )
        else:
            input_ids, label_ids, mask, global_label_ids = tuple(zip(*batch))

        data = torch.from_numpy(pad_sequences(input_ids, 0))
        # ignore_index is -1
        targets = torch.from_numpy(pad_sequences(label_ids, -1))
        mask = torch.from_numpy(pad_sequences(mask, 0))
        global_targets = torch.tensor(global_label_ids)

        return_tuple = (data, targets, mask, global_targets)
        if hasattr(self, "sample_weights"):
            sample_weights = torch.tensor(sample_weights)
            return_tuple = return_tuple + (sample_weights,)
        if hasattr(self, "kingdom_ids"):
            kingdom_ids = torch.tensor(kingdom_ids)
            return_tuple = return_tuple + (kingdom_ids,)

        return return_tuple

# 这个类定义了处理磷酸化位点文件的办法
class PhosphoicAcidThreeLineFastaDataset(Dataset):
    """Creates a dataset from a SignalP format 3-line .fasta file."""

    def __init__(
            self,
            data_path: Union[str, Path],
            tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
            add_special_tokens=False,
    ):

        super().__init__()
        # 设置是否处理特殊标记，不用处理，保持为False即可
        self.add_special_tokens = add_special_tokens

        # 如果传入的tokenizer是字符串（如默认值"iupac"），则使用TAPETokenizer进行初始化。
        # 否则，直接使用传入的标记器对象
        if isinstance(tokenizer, str):
            from tape import TAPETokenizer

            tokenizer = TAPETokenizer(vocab=tokenizer)
            # print(tokenizer.vocab)
        self.tokenizer = tokenizer

        # 初始化标签标记器，用于将标签转换为标记
        # self.label_tokenizer = SP_label_tokenizer()
        self.label_tokenizer = PhosphoicAcid_label_tokenizer()

        self.data_file = Path(data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        # 调用分离的办法并把文件中的三行给处理掉
        self.identifiers, self.sequences, self.labels = parse_threeline_fasta(
            self.data_file
        )
        # >P35583|EUKARYA|NO_SP|0，显然其第二部分是no_sp
        # 磷酸化检测不需要这样步，也就是说不需要判断类型，这一行不需要
        # self.global_labels = [x.split("|")[2] for x in self.identifiers]

    def __len__(self) -> int:
         return len(self.sequences)

    def __getitem__(self, index):
        # 获取索引为index的序列
        item = self.sequences[index]
        labels = self.labels[index]
        # 不需要global_label这个操作
        # global_label = self.global_labels[index]
        sample_weight = (
            self.sample_weights[index] if hasattr(self, "sample_weights") else None
        )
        # 不考虑界
        # kingdom_id = (
        #     SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]]
        #     if hasattr(self, "kingdom_ids")
        #     else None
        # )

        # 磷酸检测没有特殊编码，这里不需要加上
        if self.add_special_tokens == True:
            token_ids = self.tokenizer.encode(item)
        else:
            # 只进行标记化处理，然后将标记转换为ID
            # 注意调用的是tokenize去处理序列，不是label_tokenizer
            token_ids = self.tokenizer.tokenize(item)  # + [self.tokenizer.stop_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

            # print("转化得到的token_id是这样的")
            # print(type(token_ids))
            # print(token_ids)

        # 这里将label转化为id
        label_ids = self.label_tokenizer.sequence_to_token_ids(labels)
        # global_label_id = SIGNALP_GLOBAL_LABEL_DICT[global_label]

        # 创建与token_ids相同形状的输入掩码，全为1
        input_mask = np.ones_like(token_ids)

        return_tuple = (
            np.array(token_ids),
            np.array(label_ids),
            np.array(input_mask),
        )
        # 举例
        # Labels: 0000000001000
        # label_ids:  [0 0 0 0 0 0 0 0 0 1 0 0 0]
        # Input Mask: [1 1 1 1 1 1 1 1 1 1 1 1 1]

        if sample_weight is not None:
            return_tuple = return_tuple + (sample_weight,)
        # if kingdom_id is not None:
        #     return_tuple = return_tuple + (kingdom_id,)

        return return_tuple

    # 将一个批次的样本打包成一个批次张量，以便输入到模型中进行训练或推理
    # 通常会在使用 torch.utils.data.DataLoader 时作为 collate_fn 参数传递
    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        # unpack the list of tuples
        # 因为前面删除了global的部分，这里不处理
        if hasattr(self, "sample_weights"):
            input_ids, label_ids, mask, sample_weights = tuple(
                zip(*batch)
            )
        else:
            # batch中的每个元素是一个包含
            # input_ids, label_ids, mask等的元组，通过zip和tuple操作，可以得到每个属性的列表。
            input_ids, label_ids, mask = tuple(zip(*batch))

        # pad_sequences 函数对输入序列进行填充，使它们具有相同的长度。
        # torch.from_numpy 将填充后的序列从 NumPy 数组转换为 PyTorch 张量。
        # data、targets 和 mask 分别是输入 ID、标签 ID 和掩码的张量。
        data = torch.from_numpy(pad_sequences(input_ids, 0))
        # ignore_index is -1
        targets = torch.from_numpy(pad_sequences(label_ids, -1))
        mask = torch.from_numpy(pad_sequences(mask, 0))

        return_tuple = (data, targets, mask)
        return_dict = {
            "data": data,
            "targets": targets,
            "mask": mask,
        }
        if hasattr(self, "sample_weights"):
            sample_weights = torch.tensor(sample_weights)
            return_tuple = return_tuple + (sample_weights,)
            return_dict["sample_weights"] = sample_weights

        return return_tuple
        # 理论上应该返回一个字典，这里尝试一下
        # return return_dict

# 这个类可以形成分组，删除掉针对kingdom等的处理
class PartitionThreeLineFastaDatasetPA(PhosphoicAcidThreeLineFastaDataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        sample_weights_path: Union[str, Path] = None,
        tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
        partition_id: List[str] = [0, 1, 2],
        add_special_tokens=False,
        one_versus_all=False,
        positive_samples_weight=None,
        return_kingdom_ids=False,
    ):
        super().__init__(data_path, tokenizer, add_special_tokens)
        self.partition_id = partition_id
        if not one_versus_all:
            self.identifiers, self.sequences, self.labels = subset_dataset_PA(
                self.identifiers,
                self.sequences,
                self.labels,
                partition_id,
            )
        else:
            # retain all types
            self.identifiers, self.sequences, self.labels = subset_dataset_PA(
                self.identifiers,
                self.sequences,
                self.labels,
                partition_id,
            )

        # >A8FLJ3|NEGATIVE|TAT|0
        # self.global_labels = [x.split("|")[2] for x in self.identifiers]

        # 默认不添加，暂保留
        if return_kingdom_ids:
            self.kingdom_ids = [x.split("|")[1] for x in self.identifiers]

            count_dict = defaultdict(lambda: 0)
            for x in self.kingdom_ids:
                count_dict[x] += 1

            self.balanced_sampling_weights = [
                1.0 / count_dict[i] for i in self.kingdom_ids
            ]

        if sample_weights_path is not None:
            sample_weights_df = pd.read_csv(sample_weights_path, index_col=0)
            subset_ids = [x.split("|")[0].lstrip(">") for x in self.identifiers]
            df_subset = sample_weights_df.loc[subset_ids]
            self.sample_weights = list(df_subset["0"])
        elif positive_samples_weight is not None:
            # make weights from global labels
            self.sample_weights = [
                positive_samples_weight if label in ["SP", "LIPO", "TAT"] else 1
                for label in self.global_labels
            ]
        # NOTE this is just to make the training script more adaptable without having to change batch handling everytime. Always make weights,
        # decide in training script whether or not to use
        else:
            # 不是作为globa_labels来进行，而是给整个第一行添加权重
            # self.sample_weights = [1 for label in self.global_labels]
            self.sample_weights = [1 for _ in self.identifiers]

# 这个类相当于进行特殊处理，比如界、全局标签等，我删除了这些东西，暂不处理
class PartitionThreeLineFastaDataset(ThreeLineFastaDataset):
    """Creates a dataset from a SignalP format 3-line .fasta file.
    Supports extracting subsets from the input .fasta file.
    Inputs:
        data_path: path to .fasta file
        tokenizer: TAPETokenizer to convert sequences
        partition_id: integer 0-4, which partition to use
        kingdom_id: ['EUKARYA', 'ARCHAEA', 'NEGATIVE', 'POSITIVE']
        type_id: ['LIPO', 'NO_SP', 'SP', 'TAT']
        add_special_tokens: bool, allow tokenizer to add special tokens
        one_versus_all: bool, use all types (only so that i don't have to change the script, totally useless otherwise)
        positive_samples_weight: give a weight to positive samples #NOTE overrides weight file
    Attributes:
        type_id: type_id argument
        partition_id: partition_id argument
        kingdom_id: kingdom_id argument
        identifiers: fasta headers
        sequences: amino acid sequences
        labels: label sequences
        global_labels: global type label for each sequence
        kingdom_ids: kingdom_id for each sequence
        sample_weights: weight for each sequence, computed either from positive_samples weight or sample_weights_path
        balanced_sampling_weights: weights for balanced kingdom sampling, use with WeightedRandomSampler

    """

    def __init__(
        self,
        data_path: Union[str, Path],
        sample_weights_path: Union[str, Path] = None,
        tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
        partition_id: List[str] = [0, 1, 2, 3, 4],
        kingdom_id: List[str] = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
        type_id: List[str] = ["LIPO", "NO_SP", "SP", "TAT"],
        add_special_tokens=False,
        one_versus_all=False,
        positive_samples_weight=None,
        return_kingdom_ids=True,
    ):
        super().__init__(data_path, tokenizer, add_special_tokens)
        self.type_id = type_id
        self.partition_id = partition_id
        self.kingdom_id = kingdom_id
        if not one_versus_all:
            self.identifiers, self.sequences, self.labels = subset_dataset(
                self.identifiers,
                self.sequences,
                self.labels,
                partition_id,
                kingdom_id,
                type_id,
            )
        else:
            # retain all types
            self.identifiers, self.sequences, self.labels = subset_dataset(
                self.identifiers,
                self.sequences,
                self.labels,
                partition_id,
                kingdom_id,
                ["LIPO", "NO_SP", "SP", "TAT"],
            )

        # >A8FLJ3|NEGATIVE|TAT|0
        self.global_labels = [x.split("|")[2] for x in self.identifiers]

        if return_kingdom_ids:
            self.kingdom_ids = [x.split("|")[1] for x in self.identifiers]

            count_dict = defaultdict(lambda: 0)
            for x in self.kingdom_ids:
                count_dict[x] += 1

            self.balanced_sampling_weights = [
                1.0 / count_dict[i] for i in self.kingdom_ids
            ]

        if sample_weights_path is not None:
            sample_weights_df = pd.read_csv(sample_weights_path, index_col=0)
            subset_ids = [x.split("|")[0].lstrip(">") for x in self.identifiers]
            df_subset = sample_weights_df.loc[subset_ids]
            self.sample_weights = list(df_subset["0"])
        elif positive_samples_weight is not None:
            # make weights from global labels
            self.sample_weights = [
                positive_samples_weight if label in ["SP", "LIPO", "TAT"] else 1
                for label in self.global_labels
            ]
        # NOTE this is just to make the training script more adaptable without having to change batch handling everytime. Always make weights,
        # decide in training script whether or not to use
        else:
            self.sample_weights = [1 for label in self.global_labels]



EXTENDED_VOCAB = [
    "NO_SP_I",
    "NO_SP_M",
    "NO_SP_O",
    "SP_S",
    "SP_I",
    "SP_M",
    "SP_O",
    "LIPO_S",
    "LIPO_I",
    "LIPO_M",
    "LIPO_O",
    "TAT_S",
    "TAT_I",
    "TAT_M",
    "TAT_O",
    "TATLIPO_S",
    "TATLIPO_I",
    "TATLIPO_M",
    "TATLIPO_O",
    "PILIN_S",
    "PILIN_I",
    "PILIN_M",
    "PILIN_O",
]

EXTENDED_VOCAB_CS = [
    "NO_SP_I",
    "NO_SP_M",
    "NO_SP_O",
    "SP_S",
    "SP_C",
    "SP_I",
    "SP_M",
    "SP_O",
    "LIPO_S",
    "LIPO_C",
    "LIPO_I",
    "LIPO_M",
    "LIPO_O",
    "TAT_S",
    "TAT_C",
    "TAT_I",
    "TAT_M",
    "TAT_O",
]


class LargeCRFPartitionDataset(PartitionThreeLineFastaDataset):
    """Same as PartitionThreeLineFastaDataset, but converts sequence labels to be used in the large CRF setup.
    Large CRF: each SP type has own states for all possbilities, no shared non-sp states.
    Label conversion is only done in __getitem__, in order not to interfere with filtering functions.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        sample_weights_path=None,
        tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
        partition_id: List[str] = [0, 1, 2, 3, 4],
        kingdom_id: List[str] = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
        type_id: List[str] = ["LIPO", "NO_SP", "SP", "TAT", "TATLIPO", "PILIN"],
        add_special_tokens=False,
        one_versus_all=False,
        positive_samples_weight=None,
        return_kingdom_ids=False,
        make_cs_state=False,
    ):
        super().__init__(
            data_path,
            sample_weights_path,
            tokenizer,
            partition_id,
            kingdom_id,
            type_id,
            add_special_tokens,
            one_versus_all,
            positive_samples_weight,
            return_kingdom_ids,
        )
        self.label_tokenizer = SP_label_tokenizer(
            EXTENDED_VOCAB_CS if make_cs_state else EXTENDED_VOCAB
        )
        self.make_cs_state = make_cs_state

    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]
        global_label = self.global_labels[index]
        weight = self.sample_weights[index] if hasattr(self, "sample_weights") else None
        kingdom_id = (
            SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]]
            if hasattr(self, "kingdom_ids")
            else None
        )

        if self.add_special_tokens == True:
            token_ids = self.tokenizer.encode(item, kingdom_id=self.kingdom_ids[index])
        else:
            token_ids = self.tokenizer.tokenize(item)  # + [self.tokenizer.stop_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

        # dependent on the global label, convert and tokenize labels
        labels = (
            labels.replace("L", "S").replace("T", "S").replace("P", "S")
        )  # all sp-same letter, type info comes from global label in next step

        # If make_cs_state, convert position before the cs from 'S' to 'C'
        if self.make_cs_state:
            last_idx = labels.rfind("S")
            if last_idx != -1:
                l = list(labels)
                l[last_idx] = "C"
                labels = "".join(l)

        converted_labels = [global_label + "_" + lab for lab in labels]
        label_ids = self.label_tokenizer.convert_tokens_to_ids(converted_labels)
        global_label_id = SIGNALP6_GLOBAL_LABEL_DICT[global_label]

        input_mask = np.ones_like(token_ids)

        return_tuple = (
            np.array(token_ids),
            np.array(label_ids),
            np.array(input_mask),
            global_label_id,
        )

        if weight is not None:
            return_tuple = return_tuple + (weight,)
        if kingdom_id is not None:
            return_tuple = return_tuple + (kingdom_id,)

        return return_tuple

class LargeCRFPartitionDatasetPA(PartitionThreeLineFastaDatasetPA):
    def __init__(
        self,
        data_path: Union[str, Path],
        sample_weights_path=None,
        tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
        partition_id: List[str] = [0, 1, 2],
        add_special_tokens=False,
        one_versus_all=False,
        positive_samples_weight=None,
        return_kingdom_ids=False,
        make_cs_state=False,
    ):
        super().__init__(
            data_path,
            sample_weights_path,
            tokenizer,
            partition_id,
            add_special_tokens,
            one_versus_all,
            positive_samples_weight,
            return_kingdom_ids,
        )
        # self.label_tokenizer = SP_label_tokenizer(
        #     EXTENDED_VOCAB_CS if make_cs_state else EXTENDED_VOCAB
        # )
        self.label_tokenizer = PhosphoicAcid_label_tokenizer()
        self.make_cs_state = make_cs_state

    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]

        # print(f"Item: {item}")
        # print(f"Labels: {labels}")

        # global_label = self.global_labels[index]
        weight = self.sample_weights[index] if hasattr(self, "sample_weights") else None
        kingdom_id = (
            SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]]
            if hasattr(self, "kingdom_ids")
            else None
        )

        token_ids = [char for char in item]
        # print(f"Manually Tokenized Item: {token_ids}")

        if self.add_special_tokens == True:
            token_ids = self.tokenizer.encode(item, kingdom_id=self.kingdom_ids[index])
        else:
            # token_ids = self.tokenizer.tokenize(item)  # + [self.tokenizer.stop_token]
            # print(f"Tokenized Item: {token_ids}")
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)
            # print(f"Token IDs: {token_ids}")

        # 这里不清楚为什么要替换，暂先不替换
        # dependent on the global label, convert and tokenize labels
        # labels = (
        #     labels.replace("L", "S").replace("T", "S").replace("P", "S")
        # )  # all sp-same letter, type info comes from global label in next step


        # If make_cs_state, convert position before the cs from 'S' to 'C'
        if self.make_cs_state:
            last_idx = labels.rfind("S")
            if last_idx != -1:
                l = list(labels)
                l[last_idx] = "C"
                labels = "".join(l)

        # converted_labels = [global_label + "_" + lab for lab in labels]
        # label_ids = self.label_tokenizer.convert_tokens_to_ids(converted_labels)
        # global_label_id = SIGNALP6_GLOBAL_LABEL_DICT[global_label]
        label_ids = self.label_tokenizer.sequence_to_token_ids(labels)
        # print(f"Label IDs: {label_ids}")

        input_mask = np.ones_like(token_ids)
        # print(f"Input Mask: {input_mask}")

        return_tuple = (
            np.array(token_ids),
            np.array(label_ids),
            np.array(input_mask),
        )

        if weight is not None:
            return_tuple = return_tuple + (weight,)
        if kingdom_id is not None:
            return_tuple = return_tuple + (kingdom_id,)

        return return_tuple


class RegionCRFDataset(Dataset):
    """Converts label sequences to array for multi-state crf.
    data_path:              training set 3-line fasta
    sample_weights_path:    optional df with a weight for each Entry in data_path
    tokenizer:              tokenizer to use for conversion of sequences
    partition_id :          list of partition ids to use
    kingdom_id:             list of kingdom ids to use
    type_id :               list of type ids to use
    add_special_tokens:     add cls, sep tokens to sequence
    label_vocab:            str-int mapping for label sequences
    global_label_dict:      str-int mapping for global labels
    positive_samples_weight: optional weight that is returned for positive samples
                             (use for loss scaling)
    return_kingdom_ids:     deprecated, always returning kingdom id now.
    make_cs_state:          deprecated, no cs state implemented for multi-tag
    add_global_label:       add the global label as a token to the start of a sequence
    augment_data_paths:     paths to additional 3-line fasta files with augmented samples.
                            are added to the real data.
    vary_n_region: randomly make n region the first 2 or 3 residues.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        sample_weights_path=None,
        tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
        partition_id: List[str] = [0, 1, 2, 3, 4],
        kingdom_id: List[str] = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
        type_id: List[str] = ["LIPO", "NO_SP", "SP", "TAT", "TATLIPO", "PILIN"],
        add_special_tokens=False,
        label_vocab=None,
        global_label_dict=None,
        positive_samples_weight=None,
        return_kingdom_ids=False,  # legacy
        make_cs_state=False,  # legacy to not break code when just plugging in this dataset
        add_global_label=False,
        augment_data_paths: List[Union[str, Path]] = None,
        vary_n_region=False,
    ):

        super().__init__()

        # set up parameters

        self.data_file = Path(data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        self.add_special_tokens = add_special_tokens
        self.vary_n_region = vary_n_region

        self.label_tokenizer = SP_label_tokenizer()

        if isinstance(tokenizer, str):
            from tape import TAPETokenizer

            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        self.label_vocab = label_vocab  # None is fine, process_SP will use default
        self.add_global_label = add_global_label
        self.global_label_dict = (
            global_label_dict
            if global_label_dict is not None
            else SIGNALP6_GLOBAL_LABEL_DICT
        )

        self.type_id = type_id
        self.partition_id = partition_id
        self.kingdom_id = kingdom_id

        # Load and filter the data

        self.identifiers, self.sequences, self.labels = parse_threeline_fasta(
            self.data_file
        )

        if augment_data_paths is not None and augment_data_paths[0] is not None:
            for path in augment_data_paths:
                ids, seqs, labs = parse_threeline_fasta(path)
                self.identifiers = self.identifiers + ids
                self.sequences = self.sequences + seqs
                self.labels = self.labels + labs

        self.identifiers, self.sequences, self.labels = subset_dataset(
            self.identifiers,
            self.sequences,
            self.labels,
            partition_id,
            kingdom_id,
            type_id,
        )
        self.global_labels = [x.split("|")[2] for x in self.identifiers]
        self.kingdom_ids = [x.split("|")[1] for x in self.identifiers]

        # make kingdom-balanced sampling weights to use if needed
        count_dict = defaultdict(lambda: 0)
        for x in self.kingdom_ids:
            count_dict[x] += 1
        self.balanced_sampling_weights = [1.0 / count_dict[i] for i in self.kingdom_ids]

        # make sample weights for either loss scaling or balanced sampling, if none defined weight=1 for each item
        if sample_weights_path is not None:
            sample_weights_df = pd.read_csv(sample_weights_path, index_col=0)
            subset_ids = [x.split("|")[0].lstrip(">") for x in self.identifiers]
            df_subset = sample_weights_df.loc[subset_ids]
            self.sample_weights = list(df_subset["0"])
        elif positive_samples_weight is not None:
            # make weights from global labels
            self.sample_weights = [
                positive_samples_weight if label in ["SP", "LIPO", "TAT"] else 1
                for label in self.global_labels
            ]
        # NOTE this is just to make the training script more adaptable without having to change batch handling everytime. Always make weights,
        # decide in training script whether or not to use
        else:
            self.sample_weights = [1 for label in self.global_labels]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]
        global_label = self.global_labels[index]
        weight = self.sample_weights[index] if hasattr(self, "sample_weights") else None
        kingdom_id = (
            SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]]
            if hasattr(self, "kingdom_ids")
            else None
        )

        if self.add_special_tokens == True:
            token_ids = self.tokenizer.encode(
                item,
                kingdom_id=self.kingdom_ids[index],
                label_id=global_label if self.add_global_label else None,
            )

        else:
            token_ids = self.tokenizer.tokenize(item)  # + [self.tokenizer.stop_token]
            token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

        label_matrix = process_SP(
            labels,
            item,
            sp_type=global_label,
            vocab=self.label_vocab,
            stochastic_n_region_len=self.vary_n_region,
        )
        global_label_id = self.global_label_dict[global_label]

        input_mask = np.ones_like(token_ids)

        # also need to return original tags or cs
        if global_label == "NO_SP":
            cs = -1
        elif global_label == "SP":
            cs = (
                labels.rfind("S") + 1
            )  # +1 for compatibility. CS reporting uses 1-based instead of 0-based indexing
        elif global_label == "LIPO":
            cs = labels.rfind("L") + 1
        elif global_label == "TAT":
            cs = labels.rfind("T") + 1
        elif global_label == "TATLIPO":
            cs = labels.rfind("T") + 1
        elif global_label == "PILIN":
            cs = labels.rfind("P") + 1
        else:
            raise NotImplementedError(f"Unknown CS defintion for {global_label}")

        return_tuple = (
            np.array(token_ids),
            label_matrix,
            np.array(input_mask),
            global_label_id,
            cs,
        )

        if weight is not None:
            return_tuple = return_tuple + (weight,)
        if kingdom_id is not None:
            return_tuple = return_tuple + (kingdom_id,)

        return return_tuple

    # needs new collate_fn, targets need to be padded and stacked.
    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        # unpack the list of tuples
        if hasattr(self, "sample_weights") and hasattr(self, "kingdom_ids"):
            (
                input_ids,
                label_ids,
                mask,
                global_label_ids,
                cleavage_sites,
                sample_weights,
                kingdom_ids,
            ) = tuple(zip(*batch))
        elif hasattr(self, "sample_weights"):
            input_ids, label_ids, mask, global_label_ids, sample_weights, cs = tuple(
                zip(*batch)
            )
        elif hasattr(self, "kingdom_ids"):
            input_ids, label_ids, mask, global_label_ids, kingdom_ids, cs = tuple(
                zip(*batch)
            )
        else:
            input_ids, label_ids, mask, global_label_ids = tuple(zip(*batch))

        data = torch.from_numpy(pad_sequences(input_ids, 0))

        # ignore_index is -1
        targets = pad_sequences(label_ids, -1)
        targets = np.stack(targets)
        targets = torch.from_numpy(targets)
        mask = torch.from_numpy(pad_sequences(mask, 0))
        global_targets = torch.tensor(global_label_ids)
        cleavage_sites = torch.tensor(cleavage_sites)

        return_tuple = (data, targets, mask, global_targets, cleavage_sites)
        if hasattr(self, "sample_weights"):
            sample_weights = torch.tensor(sample_weights)
            return_tuple = return_tuple + (sample_weights,)
        if hasattr(self, "kingdom_ids"):
            kingdom_ids = torch.tensor(kingdom_ids)
            return_tuple = return_tuple + (kingdom_ids,)

        return return_tuple


class RegionCRFDatasetPA(Dataset):
    """
    Converts label sequences to array for multi-state crf。
    data_path:              训练集的 3 行 fasta 文件
    sample_weights_path:    可选的 df，其中包含 data_path 中每个条目的权重
    tokenizer:              用于序列转换的分词器
    原本处理数据格式为：>Q8TF40|EUKARYA|NO_SP|0
    对应为：>蛋白质唯一标识|kingdom_id|type_id|partition_id，除唯一标识外，均不使用
    partition_id:           要使用的分区 ID 列表
    kingdom_id:             要使用的界 ID 列表
    type_id:                要使用的类型 ID 列表
    add_special_tokens:     在序列中添加 cls 和 sep 标记
    label_vocab:            标签序列的字符串-整数映射
    global_label_dict:      全局标签的字符串-整数映射
    positive_samples_weight: 可选的权重，返回正样本的权重（用于损失缩放）
    return_kingdom_ids:     已弃用，现在总是返回界 ID
    make_cs_state:          已弃用，未为多标签实现 cs 状态
    add_global_label:       将全局标签作为标记添加到序列的开头
    augment_data_paths:     包含增强样本的其他 3 行 fasta 文件的路径。将添加到真实数据中。
    vary_n_region: 随机将 n 区域设为前 2 或 3 个残基。

    """
    def __init__(
        self,
        data_path: Union[str, Path],
        sample_weights_path=None,
        tokenizer: Union[str, PreTrainedTokenizer] = "iupac",
        partition_id: List[str] = [0, 1, 2, 3, 4],
        kingdom_id: List[str] = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
        type_id: List[str] = ["LIPO", "NO_SP", "SP", "TAT", "TATLIPO", "PILIN"],
        add_special_tokens=False,
        label_vocab=None,
        global_label_dict=None,
        positive_samples_weight=None,
        return_kingdom_ids=False,  # legacy
        make_cs_state=False,  # legacy to not break code when just plugging in this dataset
        add_global_label=False,
        augment_data_paths: List[Union[str, Path]] = None,
        vary_n_region=False,
    ):

        super().__init__()

        # set up parameters

        # 转化路径为Path并检查存在与否
        self.data_file = Path(data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        # 两个默认为False,也不需要修改
        self.add_special_tokens = add_special_tokens
        self.vary_n_region = vary_n_region

        # 配置label解释器，改成用于检测磷酸的
        # self.label_tokenizer = SP_label_tokenizer()
        self.label_tokenizer = PhosphoicAcid_label_tokenizer()

        # 指定氨基酸的解码器，即第二行的sequence，这里一般就使用默认的iupac即可
        if isinstance(tokenizer, str):
            from tape import TAPETokenizer

            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        # 设置标签词汇表和全局标签字典,全局标签字典不需要使用但是需要重新定义一下label_vocab
        # Process_SP定义在label_processing_utils.py文件中
        self.label_vocab = label_vocab  # None is fine, process_SP will use default
        # 默认为False，不添加
        self.add_global_label = add_global_label
        # 不初始化global_label_dict
        # self.global_label_dict = (
        #     global_label_dict
        #     if global_label_dict is not None
        #     else SIGNALP6_GLOBAL_LABEL_DICT
        # )

        # 这部分全都不用
        # self.type_id = type_id
        # self.partition_id = partition_id
        # self.kingdom_id = kingdom_id

        # Load and filter the data

        self.identifiers, self.sequences, self.labels = parse_threeline_fasta(
            self.data_file
        )

        # 增强用的，一般制定为None即可，这里就不会执行
        if augment_data_paths is not None and augment_data_paths[0] is not None:
            for path in augment_data_paths:
                ids, seqs, labs = parse_threeline_fasta(path)
                self.identifiers = self.identifiers + ids
                self.sequences = self.sequences + seqs
                self.labels = self.labels + labs

        # subset_dataset 函数的主要作用是根据分区 ID、界 ID 和类型 ID 筛选出匹配的标识符、序列和标签,我们这里不需要进行筛选，默认都设置为1
        # self.identifiers, self.sequences, self.labels = subset_dataset(
        #     self.identifiers,
        #     self.sequences,
        #     self.labels,
        #     partition_id,
        #     kingdom_id,
        #     type_id,
        # )
        # 数据的全局标签以及界标签，这里也不需要添加，删除掉
        # self.global_labels = [x.split("|")[2] for x in self.identifiers]
        # self.kingdom_ids = [x.split("|")[1] for x in self.identifiers]

        # make kingdom-balanced sampling weights to use if needed
        # count_dict = defaultdict(lambda: 0)
        # for x in self.kingdom_ids:
        #     count_dict[x] += 1
        # self.balanced_sampling_weights = [1.0 / count_dict[i] for i in self.kingdom_ids]

        # make sample weights for either loss scaling or balanced sampling, if none defined weight=1 for each item
        # 如果说提供了路径，那就对遍历进行比对，但是要考虑global的问题，这里修改为下面的默认设置为1
        # if sample_weights_path is not None:
        #     sample_weights_df = pd.read_csv(sample_weights_path, index_col=0)
        #     subset_ids = [x.split("|")[0].lstrip(">") for x in self.identifiers]
        #     df_subset = sample_weights_df.loc[subset_ids]
        #     self.sample_weights = list(df_subset["0"])
        # elif positive_samples_weight is not None:
        #     # make weights from global labels
        #     self.sample_weights = [
        #         positive_samples_weight if label in ["SP", "LIPO", "TAT"] else 1
        #         for label in self.global_labels
        #     ]
        # # NOTE this is just to make the training script more adaptable without having to change batch handling everytime. Always make weights,
        # # decide in training script whether or not to use
        # else:
        #     self.sample_weights = [1 for label in self.global_labels]
        if sample_weights_path is not None:
            sample_weights_df = pd.read_csv(sample_weights_path, index_col=0)
            subset_ids = [x.split("|")[0].lstrip(">") for x in self.identifiers]
            df_subset = sample_weights_df.loc[subset_ids]
            self.sample_weights = list(df_subset["0"])
        else:
            self.sample_weights = [1 for _ in self.identifiers]

    def __len__(self) -> int:
        return len(self.sequences)

    # 相比较SP，删除了主要删除了利用global_id处理切割位点的相关代码
    def __getitem__(self, index):
        item = self.sequences[index]
        labels = self.labels[index]
        weight = self.sample_weights[index] if hasattr(self, "sample_weights") else None

        token_ids = self.tokenizer.tokenize(item)
        token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)

        # process_PA将会默认调用处理磷酸化的逻辑
        label_matrix = process_PA(
            labels,
            item,
            vocab=self.label_vocab,
        )

        input_mask = np.ones_like(token_ids)

        return_tuple = (
            np.array(token_ids),
            label_matrix,
            np.array(input_mask),
        )

        if weight is not None:
            return_tuple = return_tuple + (weight,)

        return return_tuple

    # 定义一个新的collate_fn，但是去掉对于kingdomId与globalID的处理
    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        # unpack the list of tuples
        if hasattr(self, "sample_weights") and hasattr(self, "kingdom_ids"):
            (
                input_ids,
                label_ids,
                mask,
                global_label_ids,
                cleavage_sites,
                sample_weights,
                kingdom_ids,
            ) = tuple(zip(*batch))
        elif hasattr(self, "sample_weights"):
            input_ids, label_ids, mask, global_label_ids, sample_weights, cs = tuple(
                zip(*batch)
            )
        elif hasattr(self, "kingdom_ids"):
            input_ids, label_ids, mask, global_label_ids, kingdom_ids, cs = tuple(
                zip(*batch)
            )
        else:
            input_ids, label_ids, mask = tuple(zip(*batch))

        data = torch.from_numpy(pad_sequences(input_ids, 0))

        # ignore_index is -1
        targets = pad_sequences(label_ids, -1)
        targets = np.stack(targets)
        targets = torch.from_numpy(targets)
        mask = torch.from_numpy(pad_sequences(mask, 0))
        # global_targets = torch.tensor(global_label_ids)
        # cleavage_sites = torch.tensor(cleavage_sites)

        # return_tuple = (data, targets, mask, global_targets, cleavage_sites)
        return_tuple = (data, targets, mask)

        if hasattr(self, "sample_weights"):
            sample_weights = torch.tensor(sample_weights)
            return_tuple = return_tuple + (sample_weights,)
        if hasattr(self, "kingdom_ids"):
            kingdom_ids = torch.tensor(kingdom_ids)
            return_tuple = return_tuple + (kingdom_ids,)

        return return_tuple


# nine classes: Sec/SPI signal, Tat/SPI signal, Sec/SPII signal, outer region, inner region,
# TM in-out, TM out-in, Sec SPI/Tat SPI cleavage site and Sec/SPII cleavage site)
# and perform an affine linear transformation into four classes (Sec/SPI, Sec/SPII, Tat/SPI, Other)
EXTENDED_LABELS_SIGNALP_5 = {
    "I": 0,
    "O": 1,
    "TM_io": 2,
    "TM_oi": 3,
    "S": 4,
    "L": 5,
    "T": 6,
    "TL": 7,
    "P": 8,
    "CS_SPI": 9,
    "CS_SPII": 10,
    "CS_SPIII": 11,
}

# NOTE cannot share T token here between TATLIPO and TAT as I do in SignalP6.
# Otherwise mean probability looks the same for TAT and TATLIPO.
def convert_label_string_to_id_sequence(label_string, sp_type):
    """Convert SignalP string to tokens as defined in paper"""

    tokendict = EXTENDED_LABELS_SIGNALP_5
    # can't just use dict to encode whole thing, because
    # TM-in TM-out are the same token in label string.
    is_inside = False  # first tm region must go from outside to inside
    token_list = []
    for x in label_string:
        if x in ["S", "L", "T", "P", "I", "O"]:
            token_list.append(tokendict[x])
        if x == "M":
            token_list.append(tokendict["TM_io"] if is_inside else tokendict["TM_oi"])
            is_inside = not (is_inside)  # change current topology orientation

    # convert last SP token to CS token
    if sp_type == "SP":
        pos = label_string.rfind("S")
        token_list[pos] = tokendict["CS_SPI"]
    elif sp_type == "LIPO":
        pos = label_string.rfind("L")
        token_list[pos] = tokendict["CS_SPII"]
    elif sp_type == "TAT":
        pos = label_string.rfind("T")
        token_list[pos] = tokendict["CS_SPI"]
    elif sp_type == "TATLIPO":
        pos = label_string.rfind("T")
        token_list[pos] = tokendict["CS_SPII"]

        # convert tokenlist T tokens to TL tokens
        token_list = [tokendict["TL"] if x == tokendict["T"] else x for x in token_list]
    elif sp_type == "PILIN":
        pos = label_string.rfind("P")
        token_list[pos] = tokendict["CS_SPIII"]
    elif sp_type == "NO_SP":
        pass
    else:
        raise NotImplementedError(f"SP type {sp_type} unknown")

    return token_list


SIGNALP5_VOCAB = {
    "[PAD]": 0,
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
}

# need a different region vocab, as I/M/O all map to same token id
SIGNALP5_REGION_VOCAB = {
    "NO_SP_I": 0,
    "NO_SP_M": 1,
    "NO_SP_O": 2,
    "SP_N": 3,
    "SP_H": 4,
    "SP_C": 5,
    "SP_I": 0,
    "SP_M": 1,
    "SP_O": 2,
    "LIPO_N": 6,
    "LIPO_H": 7,
    "LIPO_CS": 8,  # conserved 2 positions before the CS are not hydrophobic,but are also not considered a c region
    "LIPO_C1": 9,  # the C in +1 of the CS
    "LIPO_I": 0,
    "LIPO_M": 1,
    "LIPO_O": 2,
    "TAT_N": 10,
    "TAT_RR": 11,  # conserved RR marks the border between n,h
    "TAT_H": 12,
    "TAT_C": 13,
    "TAT_I": 0,
    "TAT_M": 1,
    "TAT_O": 2,
    "TATLIPO_N": 14,
    "TATLIPO_RR": 15,
    "TATLIPO_H": 16,
    "TATLIPO_CS": 17,
    "TATLIPO_C1": 18,  # the C in +1 of the CS
    "TATLIPO_I": 0,
    "TATLIPO_M": 1,
    "TATLIPO_O": 2,
    "PILIN_P": 19,
    "PILIN_CS": 20,
    "PILIN_H": 21,
    "PILIN_I": 0,
    "PILIN_M": 1,
    "PILIN_O": 2,
}


class SignalP5Dataset(Dataset):
    """Creates a dataset from a SignalP format 3-line .fasta file.
    Labels and tokens processed as defined in SignalP5.0
    Optionally accepts a tokenizer. If no tokenizer is specified, uses hard-coded
    SignalP5 amino acid vocab without any special tokens added.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        partition_id: List[str] = [0, 1, 2, 3, 4],
        kingdom_id: List[str] = ["EUKARYA", "ARCHAEA", "NEGATIVE", "POSITIVE"],
        type_id: List[str] = ["LIPO", "NO_SP", "SP", "TAT", "TATLIPO", "PILIN"],
        tokenizer=None,
        return_region_labels=False,
    ):

        super().__init__()

        self.return_region_labels = return_region_labels
        self.label_dict = SIGNALP6_GLOBAL_LABEL_DICT

        self.tokenizer = tokenizer
        self.aa_dict = SIGNALP5_VOCAB

        self.data_file = Path(data_path)
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        # parse and subset
        identifiers, sequences, labels = parse_threeline_fasta(self.data_file)
        self.identifiers, self.sequences, self.labels = subset_dataset(
            identifiers, sequences, labels, partition_id, kingdom_id, type_id
        )

        self.global_labels = [x.split("|")[2] for x in self.identifiers]
        self.kingdom_ids = [x.split("|")[1] for x in self.identifiers]


    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.sequences[index]
        labels = self.labels[index]
        global_label = self.global_labels[index]

        kingdom_id = SIGNALP_KINGDOM_DICT[self.kingdom_ids[index]]
        global_label_id = self.label_dict[global_label]

        if self.tokenizer is not None:
            token_ids = self.tokenizer.encode(seq)
        else:
            token_ids = [self.aa_dict[x] for x in seq]

        input_mask = np.ones_like(token_ids)

        if self.return_region_labels:
            label_ids = process_SP(
                labels, seq, sp_type=global_label, vocab=SIGNALP5_REGION_VOCAB
            )
            cs = self._find_cs(labels, global_label)

            return_tuple = (
                np.array(token_ids),
                np.array(label_ids),
                np.array(input_mask),
                global_label_id,
                kingdom_id,
                cs,
            )

        else:
            label_ids = convert_label_string_to_id_sequence(labels, global_label)

            return_tuple = (
                np.array(token_ids),
                np.array(label_ids),
                np.array(input_mask),
                global_label_id,
                kingdom_id,
            )

        return return_tuple

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        # unpack the list of tuples
        if self.return_region_labels:
            input_ids, label_ids, mask, global_label_ids, kingdom_ids, cs = tuple(
                zip(*batch)
            )
        else:
            input_ids, label_ids, mask, global_label_ids, kingdom_ids = tuple(
                zip(*batch)
            )

        data = torch.from_numpy(pad_sequences(input_ids, 0))
        # ignore_index is -1
        targets = torch.from_numpy(pad_sequences(label_ids, -1))
        mask = torch.from_numpy(pad_sequences(mask, 0))
        global_targets = torch.tensor(global_label_ids)

        return_tuple = (data, targets, mask, global_targets)
        # if hasattr(self, "sample_weights"):
        #     sample_weights = torch.tensor(sample_weights)
        #     return_tuple = return_tuple + (sample_weights,)
        if hasattr(self, "kingdom_ids"):
            kingdom_ids = torch.tensor(kingdom_ids)
            return_tuple = return_tuple + (kingdom_ids,)
        if self.return_region_labels:
            cs = torch.tensor(cs)
            return_tuple = return_tuple + (cs,)

        return return_tuple

    @staticmethod
    def _find_cs(labels, global_label):
        """Helper fn to find CS in label string (last pos with label)"""
        if global_label == "NO_SP":
            cs = -1
        elif global_label == "SP":
            cs = (
                labels.rfind("S") + 1
            )  # +1 for compatibility. CS reporting uses 1-based instead of 0-based indexing
        elif global_label == "LIPO":
            cs = labels.rfind("L") + 1
        elif global_label == "TAT":
            cs = labels.rfind("T") + 1
        elif global_label == "TATLIPO":
            cs = labels.rfind("T") + 1
        elif global_label == "PILIN":
            cs = labels.rfind("P") + 1
        else:
            raise NotImplementedError(f"Unknown CS defintion for {global_label}")

        return cs
