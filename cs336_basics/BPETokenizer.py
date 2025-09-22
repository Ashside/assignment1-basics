import os
from typing import BinaryIO
from collections import Counter

import regex as re


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class BPETokenizer:
    def __init__(self):
        pass

    def get_new_split(self,old_split: list[bytes], pair: tuple[bytes, bytes], merged: bytes) -> list[bytes]:
        new_split = []
        i = 0
        while i < len(old_split):
            if i < len(old_split) - 1 and (old_split[i], old_split[i + 1]) == pair:
                new_split.append(merged)
                i += 2
            else:
                new_split.append(old_split[i])
                i += 1
        return new_split

    def get_bpe_train(
            self,
            input_path: str | os.PathLike,
            vocab_size: int,
            special_tokens: list[str],
            **kwargs,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 初始化vocab
        # 256个单字节 + special tokens
        # 注意是bytes([i])，传入一个可迭代对象，否则会产生i个0
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # 导入特殊符号
        special_tokens_vocab = {256 + i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
        # 合并
        vocab.update(special_tokens_vocab)

        # 编译pre_token过程中要使用的正则表达式
        pre_token_re = re.compile(PAT)

        # 编译分离特殊符号的正则表达式
        special_pat = "|".join(re.escape(t) for t in special_tokens)  # 注意转义，最终是形如 a|b|c 的形式
        special_re = re.compile(special_pat) if special_pat != "" else None

        pair_dict = PairDict()
        token_dict = TokenDict()
        merges: list[tuple[bytes, bytes]] = []

        # 初始化token_dict和pair_dict
        with open(input_path, "rb") as f:
            num_chunks = os.cpu_count()
            boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                # 注意此时chunk由bytes类型转换为str类型，可以应用正则表达式
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                # 接下来使用正则表达式进行一些处理
                # 首先，需要调整换行符号，原因在报错中可以发现
                chunk = re.sub(r"\r\n?", "\n", chunk)
                # 其次，去除其中的特殊符号，根据特殊符号进行分段
                if special_re is not None:
                    segments = special_re.split(chunk)
                else:
                    segments = [chunk]
                # 最后，对每一个segment进行预分词
                for seg in segments:
                    if not seg:
                        continue
                    for token_str_match in pre_token_re.finditer(seg):
                        token_str = token_str_match.group(0)
                        token_by = token_str.encode("utf-8")  # 转为bytes类型

                        # 更新token_dict，将token_by加入token_dict，频次计数加一，划分模式为单个字符
                        try:
                            token_dict.token2count[token_by] += 1
                        except KeyError:
                            token_split = [bytes([b]) for b in token_by]
                            token_dict.token2count[token_by] = 1
                            token_dict.token2splits[token_by] = token_split

                        # 更新pair_dict，统计token_by中相邻字符对的频次
                        token_split = token_dict.token2splits[token_by]
                        for adj_pair in zip(token_split[:-1], token_split[1:]):
                            pair_dict.add_pair(adj_pair, 1, token_by)

        while len(vocab) < vocab_size:
            max_pair = pair_dict.get_max_pair()
            merged_pair = max_pair[0] + max_pair[1]
            merges.append(max_pair)
            vocab[len(vocab)] = merged_pair
            # 对所有包含该pair的token进行更新
            # 首先根据pair_dict找到包含该pair的token
            # 注意这里使用pop而不是直接访问，是因为后续不再需要访问该pair了
            # 但是由于key同时会被pop，因此后续访问时可能会出现KeyError
            tokens_to_update = pair_dict.pair2tokens.pop(max_pair)
            # 最后，统计每个token中该pair的出现次数，注意根据划分模式进行统计
            for token in tokens_to_update:
                token_freq = token_dict.token2count[token]
                old_split = token_dict.token2splits[token]
                new_split = self.get_new_split(old_split, max_pair, merged_pair)
                # 统计旧划分模式中该pair的出现次数
                old_adj_pairs = list(zip(old_split[:-1], old_split[1:]))
                new_adj_pairs = list(zip(new_split[:-1], new_split[1:]))
                # 更新token_dict中的划分方式
                token_dict.token2splits[token] = new_split

                old_cnt = Counter(old_adj_pairs)
                new_cnt = Counter(new_adj_pairs)

                # 更新计数
                for pair, count in old_cnt.items():
                    # count是pair在该token中出现的次数，token_freq是该token在语料库中出现的次数
                    # 二者的乘积就是该pair在语料库中出现的次数
                    # 更新非合并对的频数和位置
                    try:
                        # 这里可能会出现KeyError，因为某些pair可能已经被合并掉了
                        pair_dict.discard_pair(pair, count * token_freq, token)
                    except KeyError:
                        # 已经作为max_pair被合并掉了
                        # pair2tokens会有KeyError
                        continue

                for pair, count in new_cnt.items():
                    # 与上述类似

                    pair_dict.add_pair(pair, count * token_freq, token)

        return vocab, merges


class TokenDict:
    def __init__(self):
        # 记录当前所有token的频次，注意token是bytes类型
        self.token2count = {}
        # 记录每个token当前的划分模型，初始时都是单个字符
        self.token2splits: dict[bytes, list[bytes]] = {}

class PairDict:
    def __init__(self):
        self.pair2count = {}
        self.pair2tokens: dict[tuple[bytes, bytes], set[bytes]] = {}

    def add_pair(self, pair: tuple[bytes, bytes], count: int, token: bytes):
        try:
            self.pair2count[pair] += count
            self.pair2tokens[pair].add(token)
        except KeyError:
            self.pair2count[pair] = count
            self.pair2tokens[pair] = {token}
    def discard_pair(self, pair: tuple[bytes, bytes], count: int, token: bytes):
        try:
            self.pair2count[pair] -= count
            self.pair2tokens[pair].discard(token)
        except KeyError:
            # 抛给上层处理
            raise KeyError

    def __getitem__(self, pair: tuple[bytes, bytes]) -> tuple[int, set[bytes]]:
        return self.pair2count[pair], self.pair2tokens[pair]

    def get_max_pair(self) -> tuple[bytes, bytes]:
        max_freq = max(freq for freq in self.pair2count.values())
        max_pairs = [pair for pair, freq in self.pair2count.items() if freq == max_freq]
        # 返回其中字典序最大的pair
        return max(max_pairs)


