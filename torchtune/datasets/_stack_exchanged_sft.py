# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtune.data import StackExchangedPairedTemplate
from torchtune.datasets._instruct import InstructDataset
from torchtune.datasets._preference import PreferenceDataset, PreferenceDatasetForSFT
from torchtune.modules.tokenizers import Tokenizer


def stack_exchanged_sft_dataset(
    tokenizer: Tokenizer,
    *,
    source: str = "lvwerra/stack-exchange-paired",
    max_seq_len: int = 1024,
    local_dataset: bool = False,
) -> InstructDataset:
    """
    Family of preference datasets similar to `StackExchangePaired data
    <https://huggingface.co/datasets/lvwerra/stack-exchange-paired>`_.

    Args:
        tokenizer (Tokenizer): Tokenizer used to encode data.
        source (str): path string of dataset, anything supported by Hugging Face's `load_dataset`.
        max_seq_len (int): Maximum number of tokens in the returned input and label token id lists.
            Default is 1024.

    Returns:
        PreferenceDataset: The preference dataset built from source paired data.
    """
    ds = PreferenceDatasetForSFT(
        tokenizer=tokenizer,
        source=source,
        template=StackExchangedPairedTemplate(),
        column_map={
            "prompt": "question",
            "chosen": "response_j",
            "rejected": "response_k",
        },
        max_seq_len=max_seq_len,
        local_dataset = local_dataset,
        split="train",
        data_dir="data/finetune",
    )

    return ds
