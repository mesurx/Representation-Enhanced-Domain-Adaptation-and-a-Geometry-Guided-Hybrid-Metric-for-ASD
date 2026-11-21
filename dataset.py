import torch
from torch.utils.data import Dataset
import librosa
import re
import os
import utils


def parse_domain_from_filename(filename):
    base_name = os.path.basename(filename).lower()
    if "_source_" in base_name:
        return 0
    elif "_target_" in base_name:
        return 1
    else:
        raise ValueError(f"Cannot parse domain label from filename: {filename}")

class ASDDataset(Dataset):
    def __init__(self, args, file_list: list, load_in_memory=False):
        self.file_list = file_list
        self.args = args
        self.wav2mel = utils.Wave2Mel(sr=args.sr, power=args.power,
                                      n_fft=args.n_fft, n_mels=args.n_mels,
                                      win_length=args.win_length, hop_length=args.hop_length)
        self.load_in_memory = load_in_memory
        self.data_list = [self.transform(filename) for filename in file_list] if load_in_memory else []

    def __getitem__(self, item):
        data_item = self.data_list[item] if self.load_in_memory else self.transform(self.file_list[item])
        return data_item



    def transform(self, filename):
        filename = filename.replace("\\", "/")
        machine = filename.split('/')[-3]
        id_str = re.findall('section_[0-9][0-9]', filename)[0]
        label = self.args.meta2label[machine+'-'+id_str]  # 查询字典,42类
        # 新增 domain label
        domain_label = parse_domain_from_filename(filename)
        machine_label = self.args.machine_label[machine]
        x, _ = librosa.core.load(filename, sr=self.args.sr, mono=True)
        x = x[: self.args.sr * self.args.secs]
        x_wav = torch.from_numpy(x)
        x_mel = self.wav2mel(x_wav)
        return x_wav, x_mel, label,domain_label,machine_label

    def __len__(self):
        return len(self.file_list)


"20260526改，使batchsize的一半包括源域，一半包括目标域"

import numpy as np
import math
import random
from torch.utils.data import Sampler


class BalancedDomainBatchSampler(Sampler):
    def __init__(
        self,
        file_list,
        batch_size,
        source_ratio,
        drop_last=True
    ):
        self.file_list = file_list
        self.batch_size = batch_size
        self.source_ratio = source_ratio
        self.drop_last = drop_last

        self.source_indices = []
        self.target_indices = []

        for idx, filename in enumerate(file_list):
            domain = parse_domain_from_filename(filename)

            if domain == 0:
                self.source_indices.append(idx)
            elif domain == 1:
                self.target_indices.append(idx)

        if len(self.source_indices) == 0:
            raise ValueError("No source samples found.")

        if len(self.target_indices) == 0:
            raise ValueError("No target samples found.")

        self.num_source_per_batch = int(batch_size * source_ratio)
        self.num_target_per_batch = batch_size - self.num_source_per_batch

        if self.num_source_per_batch <= 0 or self.num_target_per_batch <= 0:
            raise ValueError(
                "Each batch must contain both source and target samples. "
                "Please check batch_size and source_ratio."
            )

        # 一个 epoch 中尽量遍历完 source
        if drop_last:
            self.num_batches = len(self.source_indices) // self.num_source_per_batch
        else:
            self.num_batches = math.ceil(
                len(self.source_indices) / self.num_source_per_batch
            )

    def __iter__(self):
        source_indices = self.source_indices.copy()
        random.shuffle(source_indices)

        for batch_idx in range(self.num_batches):
            start = batch_idx * self.num_source_per_batch
            end = start + self.num_source_per_batch

            source_batch = source_indices[start:end]

            if len(source_batch) < self.num_source_per_batch:
                if self.drop_last:
                    continue
                else:
                    source_batch += random.choices(
                        self.source_indices,
                        k=self.num_source_per_batch - len(source_batch))

            # target 数量很少，所以这里使用有放回采样
            # 2026.6.6   随机抽样，一个epoch可能不能被抽到
            target_batch = random.choices(
                self.target_indices,
                k=self.num_target_per_batch
            )

            batch = source_batch + target_batch
            random.shuffle(batch)

            yield batch

    def __len__(self):
        return self.num_batches




import math
import random
import re
from collections import defaultdict

from torch.utils.data import Sampler


class BalancedLabelDomainBatchSampler2(Sampler):
    """
    构造适合 DomainAwareSupConLoss 的 batch。

    默认:
        batch_size = 32
        labels_per_batch = 8
        samples_per_class = 4
        source_per_class = 2
        target_per_class = 2

    每个 batch:
        8 个 label × 每个 label 4 个样本 = 32

    每个 label 内部:
        2 个 source + 2 个 target

    注意:
        需要 args.meta2label 用于从 filename 解析 label。
        domain 通过 parse_domain_from_filename(filename) 解析。
    """

    def __init__(
        self,
        file_list,
        batch_size=32,
        labels_per_batch=8,
        source_per_class=2,
        target_per_class=2,
        args=None,
        drop_last=True,
        allow_replace=True,
    ):
        self.file_list = file_list
        self.batch_size = batch_size
        self.labels_per_batch = labels_per_batch
        self.source_per_class = source_per_class
        self.target_per_class = target_per_class
        self.samples_per_class = source_per_class + target_per_class
        self.args = args
        self.drop_last = drop_last
        self.allow_replace = allow_replace

        if self.labels_per_batch * self.samples_per_class != self.batch_size:
            raise ValueError(
                "batch_size must equal labels_per_batch * "
                "(source_per_class + target_per_class). "
                f"Got batch_size={batch_size}, "
                f"labels_per_batch={labels_per_batch}, "
                f"samples_per_class={self.samples_per_class}."
            )

        if self.args is None:
            raise ValueError("args is required because label parsing uses args.meta2label.")

        self.label_domain_indices = defaultdict(lambda: {0: [], 1: []})

        for idx, filename in enumerate(file_list):
            label = self._parse_label_from_filename(filename)
            domain = parse_domain_from_filename(filename)

            if domain not in [0, 1]:
                continue

            self.label_domain_indices[label][domain].append(idx)

        # 只保留至少有 source 或 target 的 label
        self.labels = []
        for label, domain_dict in self.label_domain_indices.items():
            num_source = len(domain_dict[0])
            num_target = len(domain_dict[1])

            if num_source + num_target > 0:
                self.labels.append(label)

        if len(self.labels) == 0:
            raise ValueError("No valid labels found.")

        # 优先使用同时有 source 和 target 的类别
        self.labels_with_both_domains = [
            label for label in self.labels
            if len(self.label_domain_indices[label][0]) > 0
            and len(self.label_domain_indices[label][1]) > 0
        ]

        if len(self.labels_with_both_domains) == 0:
            raise ValueError("No label contains both source and target samples.")

        if len(self.labels_with_both_domains) < self.labels_per_batch:
            print(
                "[Warning] labels_with_both_domains is smaller than labels_per_batch. "
                "Labels will be sampled with replacement."
            )

        # 一个 epoch 的 batch 数量可以按 source 数量估算
        total_source = sum(
            len(self.label_domain_indices[label][0])
            for label in self.labels
        )

        source_per_batch = self.labels_per_batch * self.source_per_class

        if drop_last:
            self.num_batches = total_source // source_per_batch
        else:
            self.num_batches = math.ceil(total_source / source_per_batch)

        self.num_batches = max(1, self.num_batches)

    def _parse_label_from_filename(self, filename):
        filename = filename.replace("\\", "/")
        machine = filename.split("/")[-3]
        id_str = re.findall(r"section_[0-9][0-9]", filename)[0]
        label = self.args.meta2label[machine + "-" + id_str]
        return label

    def _sample_from_pool(self, pool, k):
        """
        从 pool 中采样 k 个 index。
        pool 数量不足时，使用有放回采样。
        """
        if len(pool) == 0:
            return []

        if len(pool) >= k:
            return random.sample(pool, k)

        if self.allow_replace:
            return random.choices(pool, k=k)

        return pool

    def __iter__(self):
        for _ in range(self.num_batches):
            # 采样 label
            if len(self.labels_with_both_domains) >= self.labels_per_batch:
                selected_labels = random.sample(
                    self.labels_with_both_domains,
                    self.labels_per_batch
                )
            else:
                selected_labels = random.choices(
                    self.labels_with_both_domains,
                    k=self.labels_per_batch
                )

            batch = []

            for label in selected_labels:
                source_pool = self.label_domain_indices[label][0]
                target_pool = self.label_domain_indices[label][1]

                source_samples = self._sample_from_pool(
                    source_pool,
                    self.source_per_class
                )
                target_samples = self._sample_from_pool(
                    target_pool,
                    self.target_per_class
                )

                # 如果某个 label 的某个 domain 为空，理论上不会发生，
                # 因为 selected_labels 来自 labels_with_both_domains。
                # 这里保留兜底逻辑。
                if len(source_samples) < self.source_per_class:
                    missing = self.source_per_class - len(source_samples)
                    fallback_pool = source_pool + target_pool
                    source_samples += random.choices(fallback_pool, k=missing)

                if len(target_samples) < self.target_per_class:
                    missing = self.target_per_class - len(target_samples)
                    fallback_pool = source_pool + target_pool
                    target_samples += random.choices(fallback_pool, k=missing)

                batch.extend(source_samples)
                batch.extend(target_samples)

            random.shuffle(batch)

            if len(batch) == self.batch_size:
                yield batch
            else:
                if not self.drop_last:
                    yield batch

    def __len__(self):
        return self.num_batches

