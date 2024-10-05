# © Copyright Technical University of Denmark
import torch
import torch.nn as nn
import sys
from .multi_tag_crf import CRF
from typing import Tuple
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers import T5Model, T5PreTrainedModel, T5Tokenizer
import re


def mask_targets_and_logits(targets, prediction_logits, mask_value=-1):
    """
    逐个处理每个序列，删除 targets 中的 mask_value，并相应地调整 prediction_logits。

    参数:
    - targets: 形状为 (batch_size, seq_length) 的张量，包含目标标签。
    - prediction_logits: 形状为 (batch_size, seq_length, num_tags) 的张量，包含预测的发射分数。
    - mask_value: 要删除的掩码值，默认为 -1。

    返回:
    - masked_targets: 删除 mask_value 后的 targets，形状为 (batch_size, new_seq_length)。
    - masked_prediction_logits: 相应调整后的 prediction_logits，形状为 (batch_size, new_seq_length, num_tags)。
    """
    batch_size, seq_length, num_tags = prediction_logits.size()

    masked_targets = []
    masked_prediction_logits = []

    for i in range(batch_size):
        # 获取当前序列的 targets 和 prediction_logits
        current_targets = targets[i]
        current_logits = prediction_logits[i]

        # 找到非 mask_value 的位置
        valid_indices = current_targets != mask_value

        # 删除 mask_value 后的 targets 和相应的 prediction_logits
        masked_targets.append(current_targets[valid_indices])
        masked_prediction_logits.append(current_logits[valid_indices])

    # 将结果转换为张量
    masked_targets = torch.stack(masked_targets)
    masked_prediction_logits = torch.stack(masked_prediction_logits)

    return masked_targets, masked_prediction_logits

class SequenceDropout(nn.Module):
    """Layer zeroes full hidden states in a sequence of hidden states"""
    # 这个类用于在训练过程中随机将输入序列的一部分隐藏状态置零，即序列Dropout,以此来防止模型对训练数据过拟合。
    # 暂时还没理解怎么用7.11

    def __init__(self, p=0.1, batch_first=True):
        super().__init__()
        # p用来控制Dropout的概率
        self.p = p
        # batch_first 参数指示输入数据的维度排列方式
        self.batch_first = batch_first

    def forward(self, x):
        # 没有训练或者概率设0，则不执行
        if not self.training or self.dropout == 0:
            return x
        # print("SeqDropout的forward开始执行")

        # 如果输入x的格式不是batch_first（即第一个维度不是批量大小），则交换第一维和第二维
        if not self.batch_first:
            x = x.transpose(0, 1)
        # make dropout mask
        mask = torch.ones(x.shape[0], x.shape[1], dtype=x.dtype).bernoulli(
            1 - self.dropout
        )  # batch_size x seq_len
        # expand
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 16)
        # multiply
        after_dropout = mask_expanded * x
        if not self.batch_first:
            after_dropout = after_dropout.transpose(0, 1)
        # print("SequenceDropout执行完毕，准备返回信息！")

        return after_dropout


class ProteinBertTokenizer:
    """Wrapper class for Huggingface BertTokenizer.
    implements an encode() method that takes care of
    - putting spaces between AAs
    - 在氨基酸之间添加空格
    - prepending the kingdom id token,if kingdom id is provided and vocabulary allows it
    - 如果提供了界ID并且词汇表允许，就添加界ID标记
    - prepending the label token, if provided and vocabulary allows it. label token is used when
      predicting the CS conditional on the known class.
    - 如果提供了标签并且词汇表允许，就添加标签标记。在已知类别的条件下预测CS时使用标签标记。
    """

    # 由于要对不同生物进行区分，这里设置了界id用于处理分词，预测磷酸化应该不需这个参数

    def __init__(self, *args, **kwargs):
        # from_pretrained用于从一个预训练的模型检查点加载分词器的配置和权重
        self.tokenizer = BertTokenizer.from_pretrained(*args, **kwargs)

    def encode(self, sequence, kingdom_id=None, label_id=None):
        # Preprocess sequence to ProtTrans format
        # 用空格的形式连接，这是ProtTrans的要求
        sequence = " ".join(sequence)

        #  'U'（硒半胱氨酸）、'Z'（赖氨酸）、'O'（羟赖氨酸）和 'B'（天冬酰胺）将其转化为X，这个需要后面检查分词器词汇表
        prepro = re.sub(r"[UZOB]", "X", sequence)

        # 界编号没有，但是有EPSD ID
        if kingdom_id is not None and self.tokenizer.vocab_size > 30:
            prepro = kingdom_id.upper() + " " + prepro
        if (
            label_id is not None and self.tokenizer.vocab_size > 34
        ):  # implies kingdom is also used.
            prepro = (
                label_id.upper().replace("_", "") + " " + prepro
            )  # HF tokenizers split at underscore, can't have taht in vocab
        return self.tokenizer.encode(prepro)

    def tokenize(self, sequence):
        # 调用 BertTokenizer 的 tokenize 方法
        return self.tokenizer.tokenize(sequence)

    def convert_tokens_to_ids(self, tokens):
        # 调用 BertTokenizer 的 convert_tokens_to_ids 方法
        return self.tokenizer.convert_tokens_to_ids(tokens)

    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        return cls(checkpoint, **kwargs)
    # 用于创建类的实例，checkpoint可以指向预训练的模型或者路径

class ProteinT5Tokenizer:
    class ProteinT5Tokenizer:
        """Wrapper class for Huggingface T5Tokenizer.
        Implements an encode() method that takes care of:
        - inserting spaces between amino acids
        - 在氨基酸之间插入空格
        - handling ambiguous amino acids by converting 'U', 'Z', 'O', 'B' to 'X'
        - 处理模糊的氨基酸，将'U'、'Z'、'O'、'B' 转换为 'X'
        """

        def __init__(self, *args, **kwargs):
            # 初始化T5分词器
            self.tokenizer = T5Tokenizer.from_pretrained(*args, **kwargs)

        def encode(self, sequence, padding='longest'):
            """
            Encodes the input protein sequence into token IDs.

            Args:
                sequence (str): The protein sequence to be encoded.
                padding (str): Padding strategy ('longest', 'max_length', etc.)

            Returns:
                input_ids (torch.Tensor): Tensor of input IDs.
                attention_mask (torch.Tensor): Tensor of attention masks.
            """
            # 在氨基酸之间插入空格
            sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))

            # 使用T5分词器对序列进行编码，并进行padding
            ids = self.tokenizer.batch_encode_plus(
                [sequence],
                add_special_tokens=True,
                padding=padding
            )
            input_ids = torch.tensor(ids['input_ids'])
            attention_mask = torch.tensor(ids['attention_mask'])

            return input_ids, attention_mask

        def tokenize(self, sequence):
            """
            Tokenizes the input protein sequence.

            Args:
                sequence (str): The protein sequence to be tokenized.

            Returns:
                tokens (List[str]): List of tokens.
            """
            # 在氨基酸之间插入空格
            sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
            return self.tokenizer.tokenize(sequence)

        def convert_tokens_to_ids(self, tokens):
            """
            Converts a list of tokens into their corresponding token IDs.

            Args:
                tokens (List[str]): List of tokens to be converted.

            Returns:
                List[int]: List of token IDs.
            """
            return self.tokenizer.convert_tokens_to_ids(tokens)

        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            return cls(checkpoint, **kwargs)
        # 用于创建类的实例，checkpoint可以指向预训练的模型或者路径

# 用于信号肽预测模型的分类标签映射列表
# 为什么要分为6个部分，36又是哪来的？6个部分猜测是5种加none
SIGNALP6_CLASS_LABEL_MAP = [
    [0, 1, 2],
    [3, 4, 5, 6, 7, 8],
    [9, 10, 11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20, 21, 22],
    [23, 24, 25, 26, 27, 28, 29, 30],
    [31, 32, 33, 34, 35, 36],
]

# 磷酸化
PHOSPHORYLATION_CLASS_LABEL_MAP = [
    [0],  # 非磷酸化
    [1],  # 磷酸化
]

class BertSequenceTaggingCRF(BertPreTrainedModel):
    # 用于序列标注和全局标签预测的模型，能够预测蛋白质序列中的信号肽区域以及全局标签
    # 也就是说能够检测出哪里是信号肽，以及这个信号肽是什么
    # 修改时删去信号肽是什么，保留检测部分，并且改为位点
    """Sequence tagging and global label prediction model (like SignalP).
    LM output goes through a linear layer with classifier_hidden_size before being projected to num_labels outputs.
    语言模型的输出在被映射到 num_labels 输出之前，先通过具有 classifier_hidden_size 大小的线性层。
    These outputs then either go into the CRF as emissions, or to softmax as direct probabilities.
    config.use_crf controls this.
    这些输出接着要么作为 CRF 的发射概率输入，要么直接通过 softmax 作为概率输出。这由 config.use_crf 控制。

    Inputs are batch first.
       Loss is sum between global sequence label crossentropy and position wise tags crossentropy.
       Optionally use CRF.

    """

    def __init__(self, config):
        super().__init__(config)

        # 由于不涉及到界，这部分可以不管
        ## Set up kingdom ID embedding layer if used
        self.use_kingdom_id = (
            config.use_kingdom_id if hasattr(config, "use_kingdom_id") else False
        )
        # 创建嵌入层
        if self.use_kingdom_id:
            self.kingdom_embedding = nn.Embedding(4, config.kingdom_embed_size)

        ## Set up LM and hidden state postprocessing
        # 初始化Bert模型
        self.bert = BertModel(config=config)
        # 初始化LM输出的dropout层，用于正则化
        self.lm_output_dropout = nn.Dropout(
            config.lm_output_dropout if hasattr(config, "lm_output_dropout") else 0
        )  # for backwards compatbility
        self.lm_output_position_dropout = SequenceDropout(
            config.lm_output_position_dropout
            if hasattr(config, "lm_output_position_dropout")
            else 0
        )
        self.kingdom_id_as_token = (
            config.kingdom_id_as_token
            if hasattr(config, "kingdom_id_as_token")
            else False
        )  # used for truncating hidden states
        self.type_id_as_token = (
            config.type_id_as_token if hasattr(config, "type_id_as_token") else False
        )

        # 设置CRF层的输入长度，这里的70是data目录下一众fasta的长度，目前的实现是在不能通过输入数据或标签来控制序列长度的情况下使用的硬编码值
        # 不知道要不要修改，因为数据不是截成70，划分的话肯定会预测不准的
        self.crf_input_length = 100
        # TODO make this part of config if needed. Now it's for cases where I don't control that via input data or labels.

        ## Hidden states to CRF emissions
        self.outputs_to_emissions = nn.Linear(
            config.hidden_size
            if self.use_kingdom_id is False
            else config.hidden_size + config.kingdFom_embed_size,
            config.num_labels,
        )


        ## Set up CRF
        self.num_global_labels = (
            config.num_global_labels
            if hasattr(config, "num_global_labels")
            else config.num_labels
        )
        self.num_labels = config.num_labels
        self.class_label_mapping = (
            config.class_label_mapping
            if hasattr(config, "class_label_mapping")
            else SIGNALP6_CLASS_LABEL_MAP
        )
        assert (
            len(self.class_label_mapping) == self.num_global_labels
        ), "defined number of classes and class-label mapping do not agree."

        self.allowed_crf_transitions = (
            config.allowed_crf_transitions
            if hasattr(config, "allowed_crf_transitions")
            else None
        )
        self.allowed_crf_starts = (
            config.allowed_crf_starts if hasattr(config, "allowed_crf_starts") else None
        )
        self.allowed_crf_ends = (
            config.allowed_crf_ends if hasattr(config, "allowed_crf_ends") else None
        )

        self.crf = CRF(
            num_tags=config.num_labels,
            batch_first=True,
            allowed_transitions=self.allowed_crf_transitions,
            allowed_start=self.allowed_crf_starts,
            allowed_end=self.allowed_crf_ends,
        )
        # Legacy, remove this once i completely retire non-mulitstate labeling
        self.sp_region_tagging = (
            config.use_region_labels if hasattr(config, "use_region_labels") else False
        )  # use the right global prob aggregation function
        self.use_large_crf = True  # legacy for get_metrics, no other use.

        ## Loss scaling parameters
        self.crf_scaling_factor = (
            config.crf_scaling_factor if hasattr(config, "crf_scaling_factor") else 1
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        kingdom_ids=None,
        input_mask=None,
        targets=None,
        targets_bitmap=None,
        global_targets=None,
        inputs_embeds=None,
        sample_weights=None,
        return_emissions=False,
        force_states=False,
    ):
        """Predict sequence features.
        Inputs:  input_ids (batch_size, seq_len)
                 kingdom_ids (batch_size) :  [0,1,2,3] for eukarya, gram_positive, gram_negative, archaea
                 targets (batch_size, seq_len). number of distinct values needs to match config.num_labels
                 global_targets (batch_size)
                 input_mask (batch_size, seq_len). binary tensor, 0 at padded positions
                 input_embeds: Optional instead of input_ids. Start with embedded sequences instead of token ids.
                 sample_weights (batch_size) float tensor. weight for each sequence to be used in cross-entropy.
                 return_emissions : return the emissions and masks for the CRF. used when averaging viterbi decoding.


        Outputs: (loss: torch.tensor)
                 global_probs: global label probs (batch_size, num_labels)
                 probs: model probs (batch_size, seq_len, num_labels)
                 pos_preds: best label sequences (batch_size, seq_len)
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if targets is not None and targets_bitmap is not None:
            raise ValueError(
                "You cannot specify both targets and targets_bitmap at the same time"
            )
        # print("BertSeqCRF的forward开始执行")
        ## Get LM hidden states
        # 获取语言模型的隐藏状态
        outputs = self.bert(
            input_ids, attention_mask=input_mask, inputs_embeds=inputs_embeds
        )  # Returns tuple. pos 0 is sequence output, rest optional.
        # 这里取得的就是由碱基得到的那些序列，后面需要进一步的处理
        sequence_output = outputs[0]
        # print(f"After BERT - sequence_output shape: {sequence_output.shape}")


        ## Remove special tokens
        # 移除特殊 token，如 BERT 的 [CLS] 和 [SEP]，并处理 padding
        # sequence_output, input_mask = self._trim_transformer_output(
        #     sequence_output, input_mask
        # )  # this takes care of CLS and SEP, pad-aware
        # print(f"After trim - sequence_output shape: {sequence_output.shape}")
        sequence_output, input_mask = self._trim_transformer_output(sequence_output, input_mask, False)
        # print(f"After trim - sequence_output shape: {sequence_output.shape}")
        # print(f"After trim - input_mask shape: {input_mask.shape}")


        # 去除界id，类别id，确保输入给CRF的比较简洁
        if self.kingdom_id_as_token:
            sequence_output = sequence_output[:, 1:, :]
            input_mask = input_mask[:, 1:] if input_mask is not None else None
        if self.type_id_as_token:
            sequence_output = sequence_output[:, 1:, :]
            input_mask = input_mask[:, 1:] if input_mask is not None else None
        # print(f"After removing tokens - sequence_output shape: {sequence_output.shape}")

        ## Trim transformer output to length of targets or to crf_input_length
        # 根据 targets 的长度或 crf_input_length 截断 transformer 输出
        # 这里的crf_input_length在init的时候被初始化为了70
        # print(f"Before truncation - sequence_output shape: {sequence_output.shape}")
        # print(f"Before truncation - input_mask shape: {input_mask.shape}")
        # print(f"Before truncation - sequence_output content: {sequence_output}")
        # print(f"Before truncation - input_mask content: {input_mask}")

        if targets is not None:
            sequence_output = sequence_output[
                :, : targets.shape[1], :
            ]  # this removes extra residues that don't go to CRF
            input_mask = (
                input_mask[:, : targets.shape[1]] if input_mask is not None else None
            )
        else:
            sequence_output = sequence_output[:, : self.crf_input_length, :]
            input_mask = (
                input_mask[:, : self.crf_input_length]
                if input_mask is not None
                else None
            )
        # print(f"After truncation - sequence_output shape: {sequence_output.shape}")
        # print(f"After truncation - input_mask shape: {input_mask.shape}")
        # print(f"After truncation - sequence_output content: {sequence_output}")
        # print(f"After truncation - input_mask content: {input_mask}")

        ## Apply dropouts
        sequence_output = self.lm_output_dropout(sequence_output)
        # print(f"After dropout - sequence_output shape: {sequence_output.shape}")

        ## Add kingdom ids
        if self.use_kingdom_id == True:
            ids_emb = self.kingdom_embedding(kingdom_ids)  # batch_size, embed_size
            ids_emb = ids_emb.unsqueeze(1).repeat(
                1, sequence_output.shape[1], 1
            )  # batch_size, seq_len, embed_size
            sequence_output = torch.cat([sequence_output, ids_emb], dim=-1)

        ## CRF emissions
        # 序列转化为输入到CRF层的概率
        prediction_logits = self.outputs_to_emissions(sequence_output)
        # print(f"Prediction logits shape: {prediction_logits.shape}")

        ## CRF
        # 根据是否有 targets，计算 CRF 层的对数似然
        if targets is not None:
            # masked_targets, masked_prediction_logits = mask_targets_and_logits(targets, prediction_logits)
            # print("原始 targets 形状:", targets.shape)
            # print("掩码后 targets 形状:", masked_targets.shape)
            # print("原始 prediction_logits 形状:", prediction_logits.shape)
            # print("掩码后 prediction_logits 形状:", masked_prediction_logits.shape)


            log_likelihood = self.crf(
                emissions=prediction_logits,
                tags=targets,
                tag_bitmap=None,
                mask=input_mask.byte(),
                reduction="mean",
            )
            neg_log_likelihood = -log_likelihood * self.crf_scaling_factor
        elif targets_bitmap is not None:

            log_likelihood = self.crf(
                emissions=prediction_logits,
                tags=None,
                tag_bitmap=targets_bitmap,
                mask=input_mask.byte(),
                reduction="mean",
            )
            neg_log_likelihood = -log_likelihood * self.crf_scaling_factor
        else:
            neg_log_likelihood = 0

        # 计算 CRF 层的边缘概率
        probs = self.crf.compute_marginal_probabilities(
            emissions=prediction_logits, mask=input_mask.byte()
        )
        # print(f"type of probs:{type(probs)}")
        # print(f"size of probs:{probs.size()}")
        # print(f"content of probs:{probs}")

        # 根据是否使用 sp_region_tagging 计算全局标签概率
        if self.sp_region_tagging:
            global_probs = self.compute_global_labels_multistate(probs, input_mask)
        else:
            global_probs = self.compute_global_labels(probs, input_mask)

        global_log_probs = torch.log(global_probs)

        preds = self.predict_global_labels(global_probs, kingdom_ids, weights=None)

        # TODO update init_states generation to new n,h,c states and actually start using it
        # from preds, make initial sequence label vector
        if force_states:
            init_states = self.inital_state_labels_from_global_labels(preds)
        else:
            init_states = None
        viterbi_paths = self.crf.decode(
            emissions=prediction_logits,
            mask=input_mask.byte(),
            init_state_vector=init_states,
        )

        # pad the viterbi paths
        # print("before执行pad the viterbi paths")
        # print(f"viterbi_path的值为：{viterbi_paths}")
        # max_pad_len = max([len(x) for x in viterbi_paths])
        max_pad_len = self.crf_input_length
        pos_preds = [x + [-1] * (max_pad_len - len(x)) for x in viterbi_paths]
        # print(f"max_pad_len:{max_pad_len},pos_preds:{pos_preds}")
        # pos_preds = [x + [-1] * (self.crf_input_length - len(x)) for x in viterbi_paths]
        #
        # # 确保所有序列的长度都等于 crf_input_length
        # pos_preds = [x[:self.crf_input_length] for x in pos_preds]
        pos_preds = torch.tensor(
            pos_preds, device=probs.device
        )  # NOTE convert to tensor just for compatibility with the else case, so always returns same type
        padded_probs = torch.nn.functional.pad(probs, (0, 0, 0, max_pad_len - probs.size(1)), value=-1)
        # outputs = (global_probs, probs, pos_preds)  # + outputs
        outputs = (neg_log_likelihood, padded_probs, pos_preds, prediction_logits)
        # print(f"调试信息,打印prediction_logits：{prediction_logits}")

        # get the losses
        losses = neg_log_likelihood

        if global_targets is not None:
            loss_fct = nn.NLLLoss(
                ignore_index=-1,
                reduction="none" if sample_weights is not None else "mean",
            )
            global_loss = loss_fct(
                global_log_probs.view(-1, self.num_global_labels),
                global_targets.view(-1),
            )

            if sample_weights is not None:
                global_loss = global_loss * sample_weights
                global_loss = global_loss.mean()

            # losses = losses + global_loss
        # print(f"the value of targets is:{targets}")
        # 这部分代码导致多返回了一个值
        # if (
        #     targets is not None
        #     or global_targets is not None
        #     or targets_bitmap is not None
        # ):
        #
        #     outputs = (losses,) + outputs  # loss, global_probs, pos_probs, pos_preds
        #     print(f"targets不为none，此时返回的outputs为：{outputs}")
        ## Return emissions
        if return_emissions:
            outputs = outputs + (
                prediction_logits,
                input_mask,
            )  # (batch_size, seq_len, num_labels)
        # print(f"return_emission的值为{return_emissions}，此时的outputs为:{outputs}")
        # print("BertSequenceTaggingCRF执行完毕，准备返回信息！")

        return outputs

    @staticmethod
    def _trim_transformer_output(hidden_states, input_mask, remove_cls_sep=True):
        """Helper function to remove CLS, SEP tokens after passing through transformer"""

        # remove CLS
        # 移除第一个时间步
        if remove_cls_sep:
            # Step 1: Remove CLS token
            hidden_states = hidden_states[:, 1:, :]
            # print(f"After removing CLS - hidden_states shape: {hidden_states.shape}")

            if input_mask is not None:
                input_mask = input_mask[:, 1:]
                # print(f"After removing CLS - input_mask shape: {input_mask.shape}")
                # print(f"After removing CLS - input_mask content: {input_mask}")

                true_seq_lens = input_mask.sum(dim=1) - 1  # -1 for SEP
                # print(f"True sequence lengths: {true_seq_lens}")

                mask_list = []
                output_list = []
                targets_list = []
                for i in range(input_mask.shape[0]):
                    mask_list.append(input_mask[i, : true_seq_lens[i]])
                    output_list.append(hidden_states[i, : true_seq_lens[i], :])
                    # print(
                    #     f"After removing SEP - hidden_states[{i}] shape: {hidden_states[i, : true_seq_lens[i], :].shape}")
                    # print(f"After removing SEP - input_mask[{i}] shape: {input_mask[i, : true_seq_lens[i]].shape}")

                mask_out = torch.nn.utils.rnn.pad_sequence(mask_list, batch_first=True)
                hidden_out = torch.nn.utils.rnn.pad_sequence(output_list, batch_first=True)

            else:
                hidden_out = hidden_states[:, :-1, :]
                mask_out = None

        else:
            hidden_out = hidden_states
            mask_out = input_mask


        # print(f"Final hidden_out shape: {hidden_out.shape}")
        # print(f"Final mask_out shape: {mask_out.shape if mask_out is not None else 'None'}")


        return hidden_out, mask_out,

    def compute_global_labels(self, probs, mask):
        """Compute the global labels as sum over marginal probabilities, normalizing by seuqence length.
        For agrregation, the EXTENDED_VOCAB indices from signalp_dataset.py are hardcoded here.
        If num_global_labels is 2, assume we deal with the sp-no sp case.
        这里就说明了global设置为2时就是处理是或不是的例子
        """
        # probs = b_size x seq_len x n_states tensor
        # Yes, each SP type will now have 4 labels in the CRF. This means that now you only optimize the CRF loss, nothing else.
        # To get the SP type prediction you have two alternatives. One is to use the Viterbi decoding,
        # if the last position is predicted as SPI-extracellular, then you know it is SPI protein.
        # The other option is what you mention, sum the marginal probabilities, divide by the sequence length and then sum
        # the probability of the labels belonging to each SP type, which will leave you with 4 probabilities.
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

        summed_probs = (probs * mask.unsqueeze(-1)).sum(
            dim=1
        )  # sum probs for each label over axis
        sequence_lengths = mask.sum(dim=1)
        global_probs = summed_probs / sequence_lengths.unsqueeze(-1)

        # aggregate
        no_sp = global_probs[:, 0:3].sum(dim=1)

        spi = global_probs[:, 3:7].sum(dim=1)

        if self.num_global_labels > 2:
            spii = global_probs[:, 7:11].sum(dim=1)
            tat = global_probs[:, 11:15].sum(dim=1)
            tat_spi = global_probs[:, 15:19].sum(dim=1)
            spiii = global_probs[:, 19:].sum(dim=1)

            # When using extra state for CS, different indexing

            # if self.num_labels == 18:
            #    spi = global_probs[:, 3:8].sum(dim =1)
            #    spii = global_probs[:, 8:13].sum(dim =1)
            #    tat = global_probs[:,13:].sum(dim =1)

            return torch.stack([no_sp, spi, spii, tat, tat_spi, spiii], dim=-1)

        else:
            return torch.stack([no_sp, spi], dim=-1)

    def compute_global_labels_multistate(self, probs, mask):
        """Aggregates probabilities for region-tagging CRF output"""
        if mask is None:
            mask = torch.ones(probs.shape[0], probs.shape[1], device=probs.device)

        summed_probs = (probs * mask.unsqueeze(-1)).sum(
            dim=1
        )  # sum probs for each label over axis
        sequence_lengths = mask.sum(dim=1)
        global_probs = summed_probs / sequence_lengths.unsqueeze(-1)

        global_probs_list = []
        for class_indices in self.class_label_mapping:
            summed_probs = global_probs[:, class_indices].sum(dim=1)
            global_probs_list.append(summed_probs)

        return torch.stack(global_probs_list, dim=-1)

        # if self.sp2_only:
        #    no_sp = global_probs[:,0:3].sum(dim=1)
        #    spii = global_probs[:,3:].sum(dim=1)
        #    return torch.stack([no_sp,spii], dim=-1)

        # else:
        #    no_sp = global_probs[:,0:3].sum(dim=1)
        #    spi = global_probs[:,3:9].sum(dim=1)
        #    spii = global_probs[:,9:16].sum(dim=1)
        #    tat = global_probs[:,16:23].sum(dim=1)
        #    lipotat = global_probs[:, 23:30].sum(dim=1)
        #    spiii = global_probs[:,30:].sum(dim=1)

        #    return torch.stack([no_sp,spi,spii,tat,lipotat,spiii], dim=-1)

    def predict_global_labels(self, probs, kingdom_ids, weights=None):
        """Given probs from compute_global_labels, get prediction.
        Takes care of summing over SPII and TAT for eukarya, and allows reweighting of probabilities."""

        if self.use_kingdom_id:
            eukarya_idx = torch.where(kingdom_ids == 0)[0]
            summed_sp_probs = probs[eukarya_idx, 1:].sum(dim=1)
            # update probs for eukarya
            probs[eukarya_idx, 1] = summed_sp_probs
            probs[eukarya_idx, 2:] = 0

        # reweight
        if weights is not None:
            probs = probs * weights
        # predict
        preds = probs.argmax(dim=1)

        return preds

    @staticmethod
    def inital_state_labels_from_global_labels(preds):

        initial_states = torch.zeros_like(preds)
        # update torch.where((testtensor==1) | (testtensor>0))[0] #this syntax would work.
        initial_states[preds == 0] = 0
        initial_states[preds == 1] = 3
        initial_states[preds == 2] = 9
        initial_states[preds == 3] = 16
        initial_states[preds == 4] = 23
        initial_states[preds == 5] = 31

        return initial_states

class T5SequenceTaggingCRF(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 由于不涉及到界，这部分可以不管
        ## Set up kingdom ID embedding layer if used
        self.use_kingdom_id = (
            config.use_kingdom_id if hasattr(config, "use_kingdom_id") else False
        )
        # 创建嵌入层
        if self.use_kingdom_id:
            self.kingdom_embedding = nn.Embedding(4, config.kingdom_embed_size)

        ## Set up LM and hidden state postprocessing
        # 初始化T5模型
        self.t5 = T5Model(config=config)
        # 初始化LM输出的dropout层，用于正则化
        self.lm_output_dropout = nn.Dropout(
            config.lm_output_dropout if hasattr(config, "lm_output_dropout") else 0
        )  # for backwards compatbility
        self.lm_output_position_dropout = SequenceDropout(
            config.lm_output_position_dropout
            if hasattr(config, "lm_output_position_dropout")
            else 0
        )
        self.kingdom_id_as_token = (
            config.kingdom_id_as_token
            if hasattr(config, "kingdom_id_as_token")
            else False
        )  # used for truncating hidden states
        self.type_id_as_token = (
            config.type_id_as_token if hasattr(config, "type_id_as_token") else False
        )

        # 设置CRF层的输入长度，这里的70是data目录下一众fasta的长度，目前的实现是在不能通过输入数据或标签来控制序列长度的情况下使用的硬编码值
        # 不知道要不要修改，因为数据不是截成70，划分的话肯定会预测不准的
        self.crf_input_length = 30
        # TODO make this part of config if needed. Now it's for cases where I don't control that via input data or labels.

        ## Hidden states to CRF emissions
        self.outputs_to_emissions = nn.Linear(
            config.hidden_size
            if self.use_kingdom_id is False
            else config.hidden_size + config.kingdFom_embed_size,
            config.num_labels,
        )

        ## Set up CRF
        self.num_global_labels = (
            config.num_global_labels
            if hasattr(config, "num_global_labels")
            else config.num_labels
        )
        self.num_labels = config.num_labels
        self.class_label_mapping = (
            config.class_label_mapping
            if hasattr(config, "class_label_mapping")
            else SIGNALP6_CLASS_LABEL_MAP
        )
        assert (
            len(self.class_label_mapping) == self.num_global_labels
        ), "defined number of classes and class-label mapping do not agree."

        self.allowed_crf_transitions = (
            config.allowed_crf_transitions
            if hasattr(config, "allowed_crf_transitions")
            else None
        )
        self.allowed_crf_starts = (
            config.allowed_crf_starts if hasattr(config, "allowed_crf_starts") else None
        )
        self.allowed_crf_ends = (
            config.allowed_crf_ends if hasattr(config, "allowed_crf_ends") else None
        )

        self.crf = CRF(
            num_tags=config.num_labels,
            batch_first=True,
            allowed_transitions=self.allowed_crf_transitions,
            allowed_start=self.allowed_crf_starts,
            allowed_end=self.allowed_crf_ends,
        )
        # Legacy, remove this once i completely retire non-mulitstate labeling
        self.sp_region_tagging = (
            config.use_region_labels if hasattr(config, "use_region_labels") else False
        )  # use the right global prob aggregation function
        self.use_large_crf = True  # legacy for get_metrics, no other use.

        ## Loss scaling parameters
        self.crf_scaling_factor = (
            config.crf_scaling_factor if hasattr(config, "crf_scaling_factor") else 1
        )

        self.init_weights()