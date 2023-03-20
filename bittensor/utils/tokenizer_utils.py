""" Utils for tokenizer equivalence checking, logit translation, etc.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch

from typing import List, Dict, Tuple, Any, Union
from transformers import PreTrainedTokenizerBase

EPSILON = 1e-40


def get_tokenizer_alignment_splits(offset_mapping: List[tuple], offset_mapping_std: List[tuple]) -> Dict[int, tuple]:
    r"""
    Calculates split depths necessary for tokens to align input offsets to standard offsets.
    Only input offsets may be split, not standard offsets, to create one-to-one, one-to-many, or many-to-one
    token alignments between input-to-standard tokenization.
    Allows for multiple depth splits on a token.
        Args:
            offset_mapping (:obj:`List[tuple]`, `required`):
                Tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...].
            offset_mapping_std (:obj:`List[tuple]`, `required`):
                Standard tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...]

        Returns:
            splits (:obj:`Dict[int, tuple]`, `required`):
                For tokens that have to be split, {Token index: (split depth 1, split depth 2, ...), ...}.
    """

    splits = {}
    idx = 0  # index of token segment (server tokenization)
    idx_std = 0  # index of token segment (standard tokenization)

    right = offset_mapping[idx][1]  # first right edge
    right_std = offset_mapping_std[idx_std][1]  # first std right edge

    while (idx + 1 < len(offset_mapping) and
           offset_mapping[idx + 1][1] == right):  # ignore overlapping tokens
        idx += 1

    while (idx_std + 1 < len(offset_mapping_std) and
           offset_mapping_std[idx_std + 1][1] == right_std):  # ignore overlapping tokens
        idx_std += 1

    segment_count = 1  # keep count of segments traversed,
    segment_count_std = 1  # to track one-to-many, many-to-one conditions

    while idx < len(offset_mapping) and idx_std < len(offset_mapping_std):
        if right < right_std:
            # Examples: [|] edge, [\] next edge, [.] split
            # (45, 49)
            # (45, 48) (48, 51) std
            #  |  .|    \
            #  |  |  |
            if segment_count == 1 and segment_count_std > 1:  # prevent many-to-many
                #  |     . |   \
                #  | | | |   |
                left = offset_mapping[idx][0]
                left_std = offset_mapping_std[idx_std][0]
                splits.setdefault(idx, [])
                splits[idx] += [left_std - left]  # server token, split depth
                segment_count_std = 1
                continue

            idx += 1
            if idx < len(offset_mapping):
                right = offset_mapping[idx][1]
                segment_count += 1

            while (idx + 1 < len(offset_mapping) and
                   offset_mapping[idx + 1][1] == right):  # ignore right-aligned overlapping tokens
                idx += 1

        elif right_std < right:
            if segment_count_std == 1 and segment_count > 1:  # prevent many-to-many
                # Examples: [|] edge, [\] next edge, [.] split
                #  | | | | . |
                #  |       |    \

                # (775, 778, 781, 788, 791)
                # (775, 782, 785, 795) std
                # |  |  |.  . |  |          allow for multiple splits on a single token
                # |      |  |         |
                left = offset_mapping[idx][0]
                splits.setdefault(idx, [])
                splits[idx] += [right_std - left]  # server token, split depth
                segment_count = 1
                segment_count_std = 0

            idx_std += 1
            if idx_std < len(offset_mapping_std):
                right_std = offset_mapping_std[idx_std][1]
                segment_count_std += 1

            while (idx_std + 1 < len(offset_mapping_std) and
                   offset_mapping_std[idx_std + 1][1] == right_std):  # ignore right-aligned overlapping tokens
                idx_std += 1

        else:  # right == right_std
            idx += 1
            if idx < len(offset_mapping):
                right = offset_mapping[idx][1]
                segment_count = 1

            idx_std += 1
            if idx_std < len(offset_mapping_std):
                right_std = offset_mapping_std[idx_std][1]
                segment_count_std = 1

            while (idx + 1 < len(offset_mapping) and
                   offset_mapping[idx + 1][1] == right):  # ignore right-aligned overlapping tokens
                idx += 1

            while (idx_std + 1 < len(offset_mapping_std) and
                   offset_mapping_std[idx_std + 1][1] == right_std):  # ignore right-aligned overlapping tokens
                idx_std += 1

            continue

    for idx in splits:
        splits[idx] = tuple(splits[idx])  # to enable hashable depths for split_map_cache keying

    return splits


def get_tokenizer_sequence_mappings(offset_mapping: List[tuple], offset_mapping_std: List[tuple]) -> List[tuple]:
    r"""
    Greedily determine the one-to-one, one-to-many, or many-to-one token alignments
    between input-to-standard tokenizations.
    Disallow many-to-many mappings, but allow for right-aligned overlapping tokens.
        Args:
            offset_mapping (:obj:`List[tuple]`, `required`):
                Tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...].
            offset_mapping_std (:obj:`List[tuple]`, `required`):
                Standard tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...]

        Returns:
            mappings (:obj:`List[tuple]`, `required`):
                List of mapping tuples:
                [tuple( right_idx, right_idx_std,
                        segment_count_base, segment_count_std_base,
                        segment_count_overlap, segment_count_std_overlap), ...]
    """
    mappings = []

    idx = 0  # index of token segment (server tokenization)
    idx_std = 0  # index of token segment (standard tokenization)

    right = offset_mapping[idx][1]  # first right edge
    right_std = offset_mapping_std[idx_std][1]  # first std right edge

    segment_count = 1  # keep count of segments traversed,
    segment_count_std = 1  # to track one-to-many, many-to-one conditions
    segment_count_overlap = 0  # keep count of overlapping segments
    segment_count_std_overlap = 0

    while (idx + 1 < len(offset_mapping) and
           offset_mapping[idx + 1][1] == right):  # ignore overlapping tokens
        idx += 1
        segment_count_overlap += 1

    while (idx_std + 1 < len(offset_mapping_std) and
           offset_mapping_std[idx_std + 1][1] == right_std):  # ignore overlapping tokens
        idx_std += 1
        segment_count_std_overlap += 1

    while idx < len(offset_mapping) and idx_std < len(offset_mapping_std):
        if right < right_std:
            if segment_count == 1 and segment_count_std > 1:
                # Examples: [|] edge, [\] next edge, [.] split
                #  |     . |   \
                #  | | | |   |
                print('Unaligned: Expected an aligned std edge.')
                print('idx, idx_std, right, right_std, segment_count, segment_count_std')
                print(idx, idx_std, right, right_std, segment_count, segment_count_std)

            idx += 1
            if idx < len(offset_mapping):
                right = offset_mapping[idx][1]
                segment_count += 1

            while (idx + 1 < len(offset_mapping) and
                   offset_mapping[idx + 1][1] == right):  # ignore overlapping tokens
                idx += 1
                segment_count_overlap += 1

        elif right_std < right:
            if segment_count_std == 1 and segment_count > 1:
                # Examples: [|] edge, [\] next edge, [.] split
                #  | | | | . |
                #  |       |    \
                print('Unaligned: Expected an aligned edge.')
                print('idx, idx_std, right, right_std, segment_count, segment_count_std')
                print(idx, idx_std, right, right_std, segment_count, segment_count_std)

            idx_std += 1
            if idx_std < len(offset_mapping_std):
                right_std = offset_mapping_std[idx_std][1]
                segment_count_std += 1

            while (idx_std + 1 < len(offset_mapping_std) and
                   offset_mapping_std[idx_std + 1][1] == right_std):  # ignore overlapping tokens
                idx_std += 1
                segment_count_std_overlap += 1

        else:  # right == right_std
            mappings += [(idx, idx_std, segment_count, segment_count_std,
                          segment_count_overlap, segment_count_std_overlap)]

            segment_count_overlap = 0
            segment_count_std_overlap = 0

            idx += 1
            if idx < len(offset_mapping):
                right = offset_mapping[idx][1]
                segment_count = 1

            idx_std += 1
            if idx_std < len(offset_mapping_std):
                right_std = offset_mapping_std[idx_std][1]
                segment_count_std = 1

            while (idx + 1 < len(offset_mapping) and
                   offset_mapping[idx + 1][1] == right):  # ignore overlapping tokens
                idx += 1
                segment_count_overlap += 1

            while (idx_std + 1 < len(offset_mapping_std) and
                   offset_mapping_std[idx_std + 1][1] == right_std):  # ignore overlapping tokens
                idx_std += 1
                segment_count_std_overlap += 1
            continue

    mappings += [(len(offset_mapping), len(offset_mapping_std), 1, 1, 0, 0)]  # validation segment

    return mappings


def get_tokenizer_depth_split_map(tokenizer: PreTrainedTokenizerBase,
                                  depths: tuple) -> List[Dict[str, torch.LongTensor]]:
    r"""
    Split individual token strings at specified depths, retokenize each resulting segment,
    keep only the first token of each segment (if there is one).
    Purpose is to provide targets for scattering probabilities when a single distribution requires a depth split.
        Args:
            tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Tokenizer.
            depths (:obj:`tuple`, `required`):
                Tuple of depths at which tokens strings will be split.

        Returns:
            split_map (:obj:`List[Dict[str, torch.LongTensor]]`, `required`):
    """
    split_map = []

    phrases = tokenizer.batch_decode(range(tokenizer.vocab_len))  # list of variable len strings (one per token)

    # first part of the phrase up to distance characters
    split_phrases = [[phrase[:depths[0]] for phrase in phrases]]
    for i in range(len(depths)-1):
        # middle parts of the phrase from distance characters to end
        split_phrases += [[phrase[depths[i]:depths[i+1]] for phrase in phrases]]
    # right part of the phrase from distance characters to end
    split_phrases += [[phrase[depths[-1]:] for phrase in phrases]]

    for i, phrases in enumerate(split_phrases):  # loop through left, middle, right phrase collections
        side_tokens = tokenizer(phrases)['input_ids']  # tokenize phrase collection
        tokens_lens = [len(p) for p in side_tokens]  # get token lengths of each phrase
        from_idx = [i for i, l in enumerate(tokens_lens) if l > 0]  # only non-zero len tokens list
        first_tokens = [side_tokens[i][0] for i in from_idx]  # collect first tokens of each tokenized phrase
        # add dict for phrase collection, mapping from original index to first tokens of tokenized phrase substrings
        split_map += [{'from': torch.tensor(from_idx, dtype=torch.long),
                       'to': torch.tensor(first_tokens, dtype=torch.long)}]

    return split_map


def split_probs(probs: torch.FloatTensor, split_map: List[Dict[str, torch.Tensor]]) -> torch.FloatTensor:
    r"""
    Split a given probability distribution over a tokenizer vocabulary, given a split_map
    of mappings from original tokens to target tokens at each depth of the split.
        Args:
            probs (:obj:`torch.FloatTensor`, `required`):
                [vocab_size] Input probability distribution over a tokenizer vocabulary.
            split_map (:obj:`List[Dict[str, torch.Tensor]]`, `required`):
                A split_map of mappings from original tokens to target tokens at each depth of the split.

        Returns:
            new_probs (:obj:`torch.FloatTensor`, `required`):
                [splits, vocab_size] A new tensor with resultant probability distribution at each index
                of the first dim, representing corresponding split depth.
    """
    splits = len(split_map)  # how many parts to the depth split map, e.g. left, middle, right parts
    vocab_size = probs.shape[0]  # retain input vocabulary size
    new_probs = torch.zeros((splits, vocab_size)).to(probs.device)  # provision prob dist for each part

    for pos in range(splits):  # loop through all parts of the split
        from_idx = split_map[pos]['from']  # from original string token index
        to_idx = split_map[pos]['to']  # to first token index of retokenized part string
        new_probs[pos].scatter_add_(0, to_idx, probs[from_idx])  # transfer probabilities to new part distributions

    return new_probs  # [splits, vocab_size]


def align_tokenizer_sequences(probs: torch.FloatTensor, offset_mapping: List[tuple], offset_mapping_std: List[tuple],
                              tokenizer: PreTrainedTokenizerBase,
                              split_map_cache: Dict[tuple, List[Dict[str, torch.Tensor]]],
                              tokens: torch.LongTensor, tokens_std: torch.LongTensor) -> Tuple[torch.FloatTensor,
                                                                                               List[tuple],
                                                                                               torch.LongTensor]:
    r"""
    Align an input tokenization distribution to standard tokenization segments by depth-splitting
    the input distribution at greedily chosen locations. Prepares the input distribution for mapping to a standard
    distribution.
        Args:
            probs (:obj:`torch.FloatTensor`, `required`):
                [sequence_len, vocab_size] Input probability distribution over a tokenizer vocabulary.
            offset_mapping (:obj:`List[tuple]`, `required`):
                Tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...].
            offset_mapping_std (:obj:`List[tuple]`, `required`):
                Standard tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...]
            tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Source tokenizer.
            split_map_cache (:obj:`Dict[tuple, List[Dict[str, torch.Tensor]]]`, `required`):
                A dictionary of depths keying split_maps of mappings from original tokens to
                target tokens at each depth of the split.
            tokens (:obj:`torch.LongTensor`, `required`):
                [sequence_len] A sequence of tokens produced by the source tokenizer.
            tokens_std (:obj:`torch.LongTensor`, `required`):
                [std_sequence_len] A sequence of tokens produced by the standard tokenizer.

        Returns:
            aligned_probs (:obj:`torch.FloatTensor`, `required`):
                [new_sequence_len, vocab_size] Aligned probability distribution over a tokenizer vocabulary.
            aligned_offset_mapping (:obj:`List[tuple]`, `required`):
                Tokenizer aligned offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...].
            aligned_tokens (:obj:`torch.LongTensor`, `required`):
                A sequence of aligned tokens produced by the source tokenizer.
    """
    aligned_tokens = []  # to store new aligned tokens
    aligned_probs = []  # to store new aligned probability distributions
    aligned_offset_mapping = []  # to store new aligned offset mappings of aligned tokens
    splits = get_tokenizer_alignment_splits(offset_mapping, offset_mapping_std)  # get necessary token split locations

    prev_idx = 0
    for idx in splits:  # each source token index that must be split
        depths = splits[idx]  # list of depths at which the token string must be split
        aligned_probs += [probs[prev_idx:idx]]  # retain preceding token probabilities
        aligned_offset_mapping += offset_mapping[prev_idx:idx]  # retain preceding offset mappings
        aligned_tokens += [tokens[prev_idx:idx]]  # retain preceding tokens

        if depths not in split_map_cache:
            # add depths split to cache to reuse in future (split map calc is relatively time-consuming)
            split_map_cache[depths] = get_tokenizer_depth_split_map(tokenizer, depths)

        new_probs = split_probs(probs[idx], split_map_cache[depths])  # [splits, vocab_size] new split probabilities
        aligned_probs += [new_probs]

        text_idx = tokenizer.decode(tokens[idx])

        # === Left part ===
        new_tokens = tokenizer(text_idx[:depths[0]], add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        aligned_tokens += [new_tokens[:1]]
        aligned_offset_mapping += [(offset_mapping[idx][0], offset_mapping[idx][0] + depths[0])]

        # === Middle parts ===
        for d in range(len(depths)-1):
            new_tokens = tokenizer(text_idx[depths[d]:depths[d+1]],
                                   add_special_tokens=False, return_tensors='pt')['input_ids'][0]
            aligned_tokens += [new_tokens[:1]]
            aligned_offset_mapping += [(offset_mapping[idx][0] + depths[d], offset_mapping[idx][0] + depths[d+1])]

        # == Right part ===
        new_tokens = tokenizer(text_idx[depths[-1]:], add_special_tokens=False, return_tensors='pt')['input_ids'][0]
        aligned_tokens += [new_tokens[:1]]
        aligned_offset_mapping += [(offset_mapping[idx][0] + depths[-1], offset_mapping[idx][1])]

        prev_idx = idx + 1

    aligned_probs += [probs[prev_idx:]]  # retain remainder of tokens probabilities
    aligned_tokens += [tokens[prev_idx:]]  # retain remainder of tokens
    aligned_offset_mapping += offset_mapping[prev_idx:]  # retain remainder of offset mappings

    aligned_probs = torch.cat(aligned_probs, dim=0)  # [sequence_len, vocab_size] assemble final probability tensor
    aligned_tokens = torch.cat(aligned_tokens, dim=0).long()  # [sequence_len] assemble final token sequence

    return aligned_probs, aligned_offset_mapping, aligned_tokens


def get_translation_map(from_tokenizer: PreTrainedTokenizerBase,
                        to_tokenizer: PreTrainedTokenizerBase) -> Dict[str, Any]:
    r"""
    Map individual token phrases from a tokenizer to another tokenizer.
        Args:
            from_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                From tokenizer.
            to_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                To tokenizer.

        Returns:
            translation_map (:obj:`Dict[str, Any]`, `required`):
                Maps for each observed length, a source token to a token sequence of that length,
                with source index to target indices.
    """
    set_vocab_len(from_tokenizer)
    set_vocab_len(to_tokenizer)

    translation_map = {'lengths': {}}

    phrases = from_tokenizer.batch_decode(range(from_tokenizer.vocab_len))  # tokens to strings

    to_tokens = to_tokenizer(phrases)['input_ids']  # convert single token from-phrases to to-tokenization
    to_tokens_lens = [len(p) for p in to_tokens]
    unique_lens = set(to_tokens_lens)
    max_len = max(unique_lens)
    counts = torch.zeros((max_len, to_tokenizer.vocab_len), dtype=torch.long)

    for l in unique_lens:  # each unique one-to-many mapping length
        from_idx = [i for i, k in enumerate(to_tokens_lens) if k == l]  # find len l to-tokenizations
        subset = [to_tokens[i] for i in from_idx]  # find len l to-tokenizations
        from_idx = torch.tensor(from_idx, dtype=torch.long)  # [subset_size]
        to_idx = torch.tensor(subset, dtype=torch.long)  # [subset_size, l]
        translation_map['lengths'][l] = {'from': from_idx,
                                         'to': to_idx}
        # accumulate counts on tokens, to be used to divide probability mass over its channeled sequences
        counts[:l, :].scatter_add_(1, to_idx.T, torch.ones((l, len(subset)), dtype=torch.long))

    translation_map['counts'] = counts
    return translation_map


def translate_one_to_many(probs_from: torch.FloatTensor, probs_to: torch.FloatTensor,
                          translation_map: Dict[str, Any]) -> None:
    r"""
    Translate a single token probability distribution from a source tokenization to a
    sequence of probability distributions over a target tokenization.
        Args:
            probs_from (:obj:`torch.FloatTensor`, `required`):
                [vocab_size] Input probability distribution over a from-tokenizer vocabulary.
            probs_to (:obj:`torch.FloatTensor`, `required`):
                [many, vocab_size] Output probability distributions over a to-tokenizer vocabulary.
            translation_map (:obj:`Dict[str, Any]`, `required`):
                Maps for each observed length, a source token to a token sequence of that length,
                with source index to target indices.

        Returns:

    """
    many_len = probs_to.shape[0]

    # === Unroll single distribution into std sequence ===
    for i in range(many_len):  # each unrolling step
        for map_len in translation_map['lengths'].keys():  # each one-to-many mapping length available
            if map_len < i + 1:
                continue  # skip unrolling steps not available in a shorter mapping length
            from_idx = translation_map['lengths'][map_len]['from']
            to_idx = translation_map['lengths'][map_len]['to'].T  # [map_len, subset_size_std]
            probs_to[i, :].scatter_add_(0, to_idx[i, :], probs_from[from_idx])  # add probs in-place


def translate_many_to_one(probs_from: torch.FloatTensor, probs_to: torch.FloatTensor,
                          translation_map: Dict[str, Any]) -> None:
    r"""
        Translate a sequence of token probability distributions from a source tokenization to a
        single token probability distribution over a target tokenization.
            Args:
                probs_from (:obj:`torch.FloatTensor`, `required`):
                    [many, vocab_size] Input probability distributions over a from-tokenizer vocabulary.
                probs_to (:obj:`torch.FloatTensor`, `required`):
                    [vocab_size] Output probability distribution over a to-tokenizer vocabulary.
                translation_map (:obj:`Dict[str, Any]`, `required`):
                    Maps for each observed length, a source token to a token sequence of that length,
                    with source index to target indices.

            Returns:

        """
    many_len = probs_from.shape[0]
    probs_from_copy = probs_from.clone()  # will modify from-probabilities

    # === Spread probability mass over realized sequences ===
    counts = translation_map['counts']  # [max_len, vocab_size]
    translation_max_len = counts.shape[0]  # maximum possible many-to-one length available in translation map

    if many_len <= translation_max_len:
        probs_from_copy /= counts[:many_len, :]  # divide probability mass by amount of paths crossing each token
    else:  # limit probs_from token depth to max_len
        probs_from_copy[:translation_max_len, :] /= counts

    # === Reverse map std token to source sequences, gather avg. sequence prob ===
    for map_len in translation_map['lengths'].keys():  # mutually exclusive over std tokens
        from_idx = translation_map['lengths'][map_len]['from']  # [subset_size_std] one std token
        to_idx = translation_map['lengths'][map_len]['to'].T  # [map_len, subset_size_std] many server token seq
        if many_len < map_len:  # sequence beyond segment_count has min probability 0
            to_idx = to_idx[:many_len, :]  # [segment_count, subset_size_std]
        server_seq_tokens = probs_from_copy.gather(1, to_idx)  # [map_len, subset_size_std] gather sequences
        probs_to[from_idx] = server_seq_tokens.sum(dim=0) / map_len  # [subset_size_std] in-place average approx.


def translate_tokenizer_probs(probs: torch.FloatTensor, probs_std: torch.FloatTensor,
                              offset_mapping: List[tuple], offset_mapping_std: List[tuple],
                              tokenizer: PreTrainedTokenizerBase, std_tokenizer: PreTrainedTokenizerBase,
                              split_map_cache: Dict[tuple, List[Dict[str, torch.Tensor]]],
                              to_translation_map: Dict[str, Any], from_translation_map: Dict[str, Any],
                              tokens: torch.LongTensor, tokens_std: torch.LongTensor) -> None:
    r"""
    Translates source token probability distributions to target probability distributions, by
    aligning segments through source token splits, then greedily performing one-to-one,
    one-to-many, many-to-one distribution mappings.
        Args:
            probs (:obj:`torch.FloatTensor`, `required`):
                [sequence_len, vocab_size] Input probability distribution over a source tokenizer vocabulary.
            probs_std (:obj:`torch.FloatTensor`, `required`):
                [std_sequence_len, std_vocab_size] Output probability distribution over a target tokenizer vocabulary.
                Reference that will be written in-place.
            offset_mapping (:obj:`List[tuple]`, `required`):
                Tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...].
            offset_mapping_std (:obj:`List[tuple]`, `required`):
                Standard tokenizer offset mappings for a specific sequence [(left_0, right_0), (left_1, right_1), ...]
            tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Source tokenizer.
            std_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Standard/target tokenizer.
            split_map_cache (:obj:`Dict[tuple, List[Dict[str, torch.Tensor]]]`, `required`):
                A dictionary of depths keying split_maps of mappings from original tokens to
                target tokens at each depth of the split. Adds split_maps to cache for faster future recall.
            tokens (:obj:`torch.LongTensor`, `required`):
                [sequence_len] A sequence of tokens produced by the source tokenizer.
            tokens_std (:obj:`torch.LongTensor`, `required`):
                [std_sequence_len] A sequence of tokens produced by the standard tokenizer.
            to_translation_map (:obj:`Dict[str, Any]`, `required`):
                Maps for each observed length, a source token to a token sequence of that length,
                with source index to target indices.
            from_translation_map (:obj:`Dict[str, Any]`, `required`):
                Maps for each observed length, a source token to a token sequence of that length,
                from target index to source indices.

        Returns:

    """
    # === Align tokenized sequences via source token splitting ===
    result = align_tokenizer_sequences(probs, offset_mapping, offset_mapping_std,
                                       tokenizer, split_map_cache, tokens.cpu(), tokens_std.cpu())
    aligned_probs, aligned_offset_mapping, aligned_tokens = result

    # === Get one-to-many / many-to-one mappings ===
    mappings = get_tokenizer_sequence_mappings(aligned_offset_mapping, offset_mapping_std)

    # === Perform probability mappings ===
    for (right_idx, right_idx_std, segment_count_base, segment_count_std_base,
         segment_count_overlap, segment_count_std_overlap) in mappings[1:]:  # don't map start token

        segment_count = segment_count_base + segment_count_overlap  # calculate effective segments length
        segment_count_std = segment_count_std_base + segment_count_std_overlap  # calculate effective segments length

        # === One-to-many / one-to-one mapping ===
        if segment_count_base == 1:
            start_idx_std = right_idx_std - segment_count_std  # calculate starting index

            translate_one_to_many(aligned_probs[right_idx-1],
                                  probs_std[start_idx_std:start_idx_std+segment_count_std],
                                  to_translation_map)

        # === Many-to-one mapping ===
        elif segment_count_std_base == 1:  # many-to-one
            start_idx = right_idx - segment_count  # calculate starting index

            translate_many_to_one(aligned_probs[start_idx:right_idx],
                                  probs_std[right_idx_std-1],
                                  from_translation_map)

        else:
            print('Undefined mapping.')


def get_top_probs(probs: torch.FloatTensor, tokenizer: PreTrainedTokenizerBase, amount: int = 10) -> str:
    r"""
    Constructs output string with top amount of highest probability token strings.
    Used to display the top probabilities.
        Args:
            probs (:obj:`torch.FloatTensor`, `required`):
                [vocab_size] Probability distribution over a tokenizer vocabulary.
            tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Tokenizer.
            amount: (:obj:`int`, `optional`):
                Amount of top tokens to return

        Returns:
            string (:obj:`str`, `required`):
            Highest probability token strings, prob[token-string] ...
    """
    string = ''

    vals, indices = probs.sort(dim=-1, descending=True)  # descending sort token probabilities

    for i in range(amount):
        string += '%.4f[%s] ' % (vals[i], tokenizer.decode(indices[i]))  # prob[token-string]

    return string


def translate_logits_to_probs_std(logits: torch.FloatTensor,
                                  offset_mapping: List[List[tuple]], offset_mapping_std: List[List[tuple]],
                                  tokenizer: PreTrainedTokenizerBase, std_tokenizer: PreTrainedTokenizerBase,
                                  split_map_cache: Dict[tuple, List[Dict[str, torch.Tensor]]],
                                  to_translation_map: Dict[str, Any], from_translation_map: Dict[str, Any],
                                  tokens: torch.LongTensor, tokens_std: torch.LongTensor,
                                  skip_equivalent: bool = True) -> torch.FloatTensor:
    r"""
        Translates source token logit scores to probability distributions over the standard tokenizer.
            Args:
                logits (:obj:`torch.FloatTensor`, `required`):
                    [batch_size, sequence_len, vocab_size] Input source logits over a source tokenizer vocabulary.
                offset_mapping (:obj:`List[List[tuple]]`, `required`):
                    Batch of tokenizer offset mappings
                    [[(left_0, right_0), (left_1, right_1), ...], ...].
                offset_mapping_std (:obj:`List[List[tuple]]`, `required`):
                    Batch of standard tokenizer offset mappings
                    [[(left_0, right_0), (left_1, right_1), ...], ...].
                tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    Source tokenizer.
                std_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                    Standard/target tokenizer.
                split_map_cache (:obj:`Dict[tuple, List[Dict[str, torch.Tensor]]]`, `required`):
                    A dictionary of depths keying split_maps of mappings from original tokens to
                    target tokens at each depth of the split. Adds split_maps to cache for faster future recall.
                tokens (:obj:`torch.LongTensor`, `required`):
                    [batch_size, sequence_len] A sequence of tokens produced by the source tokenizer.
                tokens_std (:obj:`torch.LongTensor`, `required`):
                    [batch_size, std_sequence_len] A sequence of tokens produced by the standard tokenizer.
                to_translation_map (:obj:`Dict[str, Any]`, `required`):
                    Maps for each observed length, a source token to a token sequence of that length,
                    with source index to target indices.
                from_translation_map (:obj:`Dict[str, Any]`, `required`):
                    Maps for each observed length, a source token to a token sequence of that length,
                    from target index to source indices.
                skip_equivalent (:obj:`bool`, `optional`):
                    Skips translation if tokenizer and std_tokenizer are equivalent.

            Returns:
                probs_std (:obj:`torch.FloatTensor`, `required`):
                    [batch_size, std_sequence_len, std_vocab_size] Output probability distribution over the
                    standard tokenizer vocabulary.
        """
    set_vocab_len(tokenizer)
    set_vocab_len(std_tokenizer)

    # === Check tokenizer equivalence / Skip if equivalent ===
    if skip_equivalent and check_tokenizer_equivalence(tokenizer, std_tokenizer):
        logits = logits.to(torch.float).to('cpu')
        probs = torch.softmax(logits, dim=2)
        return probs

    # === Get shape sizes ===
    batch_size, sequence_len, vocab_size = logits.shape
    std_sequence_len = tokens_std.shape[-1]
    std_vocab_size = std_tokenizer.vocab_len

    if tokenizer.vocab_len < vocab_size:
        logits = logits[..., :tokenizer.vocab_len]
        vocab_size = tokenizer.vocab_len

    # === Convert logits to probabilities ===
    logits = logits.to(torch.float).to('cpu')
    probs = torch.softmax(logits, dim=2)  # [batch_size, sequence_len, vocab_size]

    if vocab_size < tokenizer.vocab_len:  # fixes bug when model logits output is not full width
        padded_probs = torch.zeros((batch_size, sequence_len, tokenizer.vocab_len))
        padded_probs[..., :vocab_size] = probs
        probs = padded_probs

    # === Translate to probabilities over standard tokenizer ===
    probs_std = torch.zeros(batch_size, std_sequence_len, std_vocab_size)
    for b in range(batch_size):
        probs_b = probs[b][-len(offset_mapping[b]):]  # remove left padding
        tokens_b = tokens[b][-len(offset_mapping[b]):]  # remove left padding
        translate_tokenizer_probs(probs_b, probs_std[b], offset_mapping[b], offset_mapping_std[b],
                                  tokenizer, std_tokenizer,
                                  split_map_cache, to_translation_map, from_translation_map,
                                  tokens_b, tokens_std[b])

    # === Correct excess probability mass (haircut) ===
    probs_std_sum = probs_std.sum(dim=-1)  # [batch_size, std_sequence_len]
    over = (probs_std_sum > 1)
    probs_std[over] /= probs_std_sum[over][:, None]

    # === Correct deficient probability mass (raise) ===
    probs_std_sum = probs_std.sum(dim=-1)  # [batch_size, std_sequence_len]
    under = (probs_std_sum < 1)
    probs_std[under] += ((1 - probs_std_sum[under]) / probs_std[under].shape[-1])[:, None]  # raise noise floor so sum 1

    return probs_std  # [batch_size, std_sequence_len, std_vocab_size]


def topk_token_phrases(logits: torch.Tensor, tokenizer: PreTrainedTokenizerBase,
                       topk: int, ignore_index: int = -100) -> torch.Tensor:
    r"""
    Select topk tokenizer logits/phrases and include std_token_phrases counterparts (std_tokenization of token text)
    in topk_tensor output of shape [batch_size, (topk + 1), max_len], where max len of all phrase lists
    (with prob in front) is max_{b,k}(len([prob_k, tok_0_k, tok_1_k, ...])).
    The output topk_tensor also includes a floor_prob for each batch item. The floor probability is the
    mean probability of token phrases not captured in topk, required since the tokenizer vocab_size may
    not be known to the receiver.
    Requires prep_tokenizer(tokenizer, std_tokenizer) to set_std_token_phrases first, to make
    std_token_phrases available here.
        Args:
            logits (:obj:`torch.Tensor`, `required`):
                [batch_size, vocab_size] Input source logits for last token over a source tokenizer vocabulary.
            tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Source tokenizer (usually server tokenizer)
            topk (:obj:`int`, `required`):
                Amount of top phrases to expect (to check for mismatch)
            ignore_index (:obj:`int`, `optional`):
                Padding value to use for unfilled token positions in a shorter token phrase.

        Returns:
            topk_tensor (:obj:`torch.Tensor`, `required`):
                [batch_size, (topk + 1), max_len] tensor includes topk token probabilities (prob_k) + floor_prob
                in first column with gradients attached, with std_tokens in remaining columns with ignore_index padding.
                Content structure:
                [[[prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., ignore_index?],
                  [prob_k=1_b=0, tok_0_k=1_b=0, tok_1_k=1_b=0, ..., ignore_index?],
                  [...],
                  [prob_floor_b=0, ignore_index, ..., ignore_index]],
                 [[prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., ignore_index?],
                  [prob_k=1_b=1, tok_0_k=1_b=1, tok_1_k=1_b=1, ..., ignore_index?],
                  [...],
                  [prob_floor_b=1, ignore_index, ..., ignore_index]],
                 [...]]
    """
    # Get shape sizes
    batch_size, vocab_size = logits.shape  # [batch_size, vocab_size] only last token prediction

    # Convert logits to probabilities
    logits = logits.float()  # ensure further computations done in float32 for improved precision
    probs = torch.softmax(logits, dim=1)  # [batch_size, vocab_size]

    # TopK phrase selection
    topk_probs, topk_indices = torch.topk(probs, topk)  # topk probs and indices: [batch_size, topk]

    # === Calculate floor probability ===
    topk_pmass = topk_probs.sum(dim=-1)  # [batch_size] topk probability mass
    remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # [batch_size] remainder probability mass
    floor_probs = remainder_pmass / (vocab_size - topk)  # [batch_size]divide remainder

    # convert to list for faster iteration in list comprehension
    topk_probs_list = topk_probs.tolist()
    topk_indices_list = topk_indices.tolist()
    floor_probs_list = floor_probs.tolist()

    # === Construct topk phrases list ===
    probs = []  # collect probability tensors with gradients attached (to be grafted into topk_tensor)
    phrases = []  # form topk token phrases with prob prepend [prob, tok_0, tok_1, ... tok_n]

    for b in range(batch_size):
        # collect probability tensors with gradients attached (to be grafted into topk_tensor)
        probs += [topk_probs[b], floor_probs[b]]  # [tensor(prob_k=0_b, prob_k=1_b, ...), tensor(prob_floor_b)]

        # form topk token phrases with prob prepend [prob, tok_0, tok_1, ... tok_n]
        phrases += [[prob] + tokenizer.std_token_phrases[i]
                    for prob, i in zip(topk_probs_list[b], topk_indices_list[b])]  # [prob_k, tok_0_k, tok_1_k, ...]

        # also add prob_floor for batch item
        phrases += [[floor_probs_list[b]]]  # [prob_floor_b]

    # determine width of topk_tensor as max len of all phrase lists (with prob in front)
    max_len = max([len(p) for p in phrases])  # max_{b,k}(len([prob_k, tok_0_k, tok_1_k, ...]))

    # form single 2D tensor with all phrase and probs (typically to send to axon wire encoding)
    topk_tensor = torch.tensor([p + [ignore_index] * (max_len - len(p))
                                for p in phrases]).to(logits.device)  # [batch_size * (topk + 1), max_len]

    # grafting probability tensors into first column to attach gradients
    topk_tensor[:, 0] = torch.hstack(probs)  # tensor([prob_k=0_b, prob_k=1_b, ..., prob_floor_b])

    topk_tensor = topk_tensor.reshape(batch_size, topk + 1, max_len)  # [batch_size, (topk + 1), max_len] reshaped

    return topk_tensor  # [batch_size, (topk + 1), max_len] (probability gradients attached in first column)


def compact_topk_token_phrases(topk_tensor: torch.Tensor):
    r"""
    Compact 2D topk_tensor [batch_size, (topk + 1), max_len] by removing ignore_index padding, and also offset
    tokens by 2 to preserve [0, 1] for probabilities to allow for proper unraveling demarcated by
    probability boundaries.
        Args:
            topk_tensor (:obj:`torch.Tensor`, `required`):
                [batch_size, (topk + 1), max_len] tensor includes topk token probabilities (prob_k) + floor_prob
                in first column with gradients attached, with std_tokens in remaining columns with ignore_index padding.
                Content structure:
                [[[prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., ignore_index?],
                  [prob_k=1_b=0, tok_0_k=1_b=0, tok_1_k=1_b=0, ..., ignore_index?],
                  [...],
                  [prob_floor_b=0, ignore_index, ..., ignore_index]],
                 [[prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., ignore_index?],
                  [prob_k=1_b=1, tok_0_k=1_b=1, tok_1_k=1_b=1, ..., ignore_index?],
                  [...],
                  [prob_floor_b=1, ignore_index, ..., ignore_index]],
                 [...]]

        Returns:
            compact_topk (:obj:`torch.Tensor`, `required`):
                [sum_b(sum_k(len(phrase_k) + 1)_b)] Compacted 1-D tensor >= batch_size * (2 * topk + 1),
                since 2 * topk + 1: topk x [probability, token sequence (at least one token)] +
                floor probability (rest).
                Content structure:
                    [prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., prob_k=1_b=0, tok_0_k=1_b=0, ..., prob_floor_b=0,
                     prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., prob_k=1_b=1, tok_0_k=1_b=1, ..., prob_floor_b=1,
                     ...]
    """
    topk_tensor_offset = topk_tensor.clone()  # assume topk_tensor may be reused elsewhere so clone
    topk_tensor_offset[:, :, 1:] += 2  # add 2 to token ids to preserve [0, 1] for probabilities (in first column)

    flattened = topk_tensor_offset.flatten()  # [batch_size * (topk + 1) * max_len] 1D tensor
    compact_topk = flattened[flattened > -1]  # remove ignore_index < -1 padding to compact content

    return compact_topk  # [>= batch_size * (2 * topk + 1)]


def unravel_topk_token_phrases(compact_topk: torch.Tensor, topk: int, ignore_index: int = -100) -> torch.Tensor:
    r"""
    Unravel topk token phrases input_tensor from 1-D to [batch_size, (topk + 1), max_len] topk_tensor, which
    includes topk token probabilities (prob_k) + floor_prob in first column with gradients attached, with
    std_tokens in remaining columns with ignore_index padding.
        Args:
            compact_topk (:obj:`torch.Tensor`, `required`):
                [sum_b(sum_k(len(phrase_k) + 1)_b)] Compacted 1-D tensor >= batch_size * (2 * topk + 1),
                since 2 * topk + 1: topk x [probability, token sequence (at least one token)] +
                floor probability (rest).
                Content structure:
                    [prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., prob_k=1_b=0, tok_0_k=1_b=0, ..., prob_floor_b=0,
                     prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., prob_k=1_b=1, tok_0_k=1_b=1, ..., prob_floor_b=1,
                     ...]
            topk (:obj:`int`, `required`):
                Amount of top phrases to expect (to check for mismatch)
            ignore_index (:obj:`int`, `optional`):
                Padding value to use for unfilled token positions in a shorter token phrase.
        Returns:
            topk_tensor (:obj:`torch.Tensor`, `required`):
                [batch_size, (topk + 1), max_len] tensor includes topk token probabilities (prob_k) + floor_prob
                in first column with gradients attached, with std_tokens in remaining columns with ignore_index padding.
                Content structure:
                [[[prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., ignore_index?],
                  [prob_k=1_b=0, tok_0_k=1_b=0, tok_1_k=1_b=0, ..., ignore_index?],
                  [...],
                  [prob_floor_b=0, ignore_index, ..., ignore_index]],
                 [[prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., ignore_index?],
                  [prob_k=1_b=1, tok_0_k=1_b=1, tok_1_k=1_b=1, ..., ignore_index?],
                  [...],
                  [prob_floor_b=1, ignore_index, ..., ignore_index]],
                 [...]]
    """

    atol = 1e-6  # absolute tolerance
    # Find probability markers (per batch item: topk phrase probabilities + floor_prob)
    prob_idx = torch.where((-atol < compact_topk) & (compact_topk < 1 + atol))[0]  # 0 <= prob <= 1 [batch_size * (topk + 1)], expect token_ids >= 2

    batch_size = len(prob_idx) // (topk + 1)  # (batch_size * (topk + floor)) / (topk + floor)
    assert batch_size * (topk + 1) == len(prob_idx), f'unravel_topk_token_phrases() probability marker failure: ' \
                                                     f'{batch_size} * ({topk} + 1) != {len(prob_idx)}'  # decoding irregularity otherwise

    probs = torch.clamp(compact_topk[prob_idx], 0, 1)  # [batch_size * (topk + 1)] ensure probabilities within [0, 1]
    probs_sum = probs.reshape(batch_size, topk + 1).sum(dim=1)  # [batch_size]
    assert torch.all((-atol < probs_sum) & (probs_sum < 1 + atol)), f'unravel_topk_token_phrases(): probs_sum not in [0, 1]'

    # Obtain phrase lengths and maximum phrase length
    phrase_len = prob_idx[1:] - prob_idx[:-1]  # [batch_size * (topk + 1) - 1] length of each phrase
    phrase_len = torch.cat((phrase_len, torch.tensor([1])))  # [batch_size * (topk + 1)] prob_floor is always len=1
    max_len = phrase_len.max()  # determine width of topk_tensor as max len of all phrase lists (with prob in front)

    # Initialize topk_tensor with ignore_index + 2, since decrement with 2 follows to remove token offset later
    topk_tensor = torch.ones((batch_size * (topk + 1), max_len), device=compact_topk.device)
    topk_tensor *= ignore_index + 2  # [batch_size * (topk + 1), max_len]

    # Insert phrases of each unique length as block into topk_tensor
    for unique_len in phrase_len.unique():
        if unique_len <= 1:
            continue  # skip probability column, will be added afterward

        phrase_idx = torch.where(phrase_len == unique_len)[0]  # phrase indices where phrase_len is unique_len
        compact_idx = prob_idx[phrase_idx]  # indices in compact_topk

        # Create indexing block, add index for each phrase position, skip first (prob) position
        block_idx = [compact_idx + position for position in range(1, unique_len)]  # incrementally add each position of phrase
        # transpose .t() ensures correct interleaving of consecutive positions:
        # [[phrase_a_1, phrase_a_2, ..., phrase_a_n], [phrase_b_1, phrase_b_2, ..., phrase_b_n], ...]
        block_idx = torch.vstack(block_idx).t().reshape(-1, unique_len - 1)  # [-1, unique_len - 1] for all phrases with unique_len

        topk_tensor[phrase_idx, 1:unique_len] = compact_topk[block_idx]  # slice selected phrases and copy into topk_tensor

    topk_tensor -= 2  # remove token offset, overwrites probability column, replace probabilities below

    # grafting probability tensors into first column to attach gradients
    topk_tensor[:, 0] = probs  # tensor([prob_k=0_b, prob_k=1_b, ..., prob_floor_b])

    topk_tensor = topk_tensor.reshape(batch_size, topk + 1, max_len)  # [batch_size, (topk + 1), max_len] reshaped

    return topk_tensor  # [batch_size, (topk + 1), max_len]


def phrase_cross_entropy(target_phrases: Union[List[List[int]], torch.Tensor],
                         topk_tensor: torch.Tensor,
                         ignore_index: int = -100, reduce=True, reduction='mean',
                         vocab_size_min: int = 50257) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Calculates the cross entropy of a phrase prediction against a target phrase, so that this is a multi-token
    extension of typical cross entropy calculated for next token prediction.
        Args:
            target_phrases (:obj:`List[List[int]]`, `required`):
                [batch_size, *] Target phrases in standard token sequence list.
            topk_tensor (:obj:`torch.Tensor`, `required`):
                [batch_size, (topk + 1), max_len] tensor includes topk token probabilities (prob_k) + floor_prob
                in first column with gradients attached, with std_tokens in remaining columns with ignore_index padding.
                Content structure:
                [[[prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., ignore_index?],
                  [prob_k=1_b=0, tok_0_k=1_b=0, tok_1_k=1_b=0, ..., ignore_index?],
                  [...],
                  [prob_floor_b=0, ignore_index, ..., ignore_index]],
                 [[prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., ignore_index?],
                  [prob_k=1_b=1, tok_0_k=1_b=1, tok_1_k=1_b=1, ..., ignore_index?],
                  [...],
                  [prob_floor_b=1, ignore_index, ..., ignore_index]],
                 [...]]
            ignore_index (:obj:`int`, `optional`):
                Padding value to use for unfilled token positions in a shorter token phrase.
            reduce (:obj:`bool`, `optional`):
                Whether to reduce the cross entropy over the batch dimension.
            reduction (:obj:`str`, `optional`):
                Reduction function to perform when reduce is True.
            vocab_size_min (:obj:`int`, `optional`):
                Minimum server vocab_size expected, should set to nominal 50257,
                used to prevent the floor_probs from being too large.
        Returns:
            loss_val (:obj:`torch.Tensor`, `required`):
                Validation cross entropy loss, either scalar if reduce or [batch_size].
            loss (:obj:`torch.Tensor`, `required`):
                Phrase cross entropy loss, either scalar if reduce or [batch_size].
    """

    batch_size, topk_p1, max_len = topk_tensor.shape  # [batch_size, (topk + 1), max_len]
    topk = topk_p1 - 1

    topk_tokens = topk_tensor[:, :-1, 1:].round().int()  # [batch_size, topk, max_len - 1] Phrase tokens with ignore_index token for padding.
    topk_probs = topk_tensor[:, :-1, 0]  # [batch_size, topk] Probabilities for each phrase in topk
    floor_probs = topk_tensor[:, -1, 0]  # [batch_size] Floor probabilities as mean probability for non-topk tokens

    topk_probs = torch.clamp(topk_probs, 0, 1)  # [batch_size, topk] ensure probabilities within [0, 1]
    floor_probs = torch.clamp(floor_probs, 0, 1)  # [batch_size] ensure floor probabilities within [0, 1]

    # === Ensure total probability is 1 ===
    total_probs = topk_probs.sum(dim=-1) + max(0, vocab_size_min - topk) * floor_probs  # [batch_size] total probs
    n_topk_probs = topk_probs / total_probs[:, None]  # [batch_size, topk] normalized topk_probs
    n_floor_probs = floor_probs / total_probs  # [batch_size] normalized floor_probs

    val_probs = torch.zeros(batch_size).to(topk_probs.device)  # accumulate probabilities when first tokens match
    match_probs = torch.zeros(batch_size).to(topk_probs.device)  # accumulate probabilities when sub target matches phrase
    for b in range(batch_size):
        target_phrase = target_phrases[b]
        if not isinstance(target_phrase, torch.Tensor):
            target_phrase = torch.tensor(target_phrases[b])
        if isinstance(target_phrase, torch.FloatTensor):
            target_phrase = target_phrase.round().int()

        match = (topk_tokens[b, :, 0] == target_phrase[0].item())  # bool where first tokens match (validation token)
        if match.sum() > 0:
            val_probs[b] = n_topk_probs[b, match].sum()  # accumulate all matches
        else:  # no matches
            val_probs[b] = n_floor_probs[b]  # assume match is in non-topk tokens with avg floor_prob

        # === Integrate sub target matches ===
        check_len = min(max_len - 1, len(target_phrase))
        for c in range(1, check_len + 1):  # progressively increase sub target length
            target = ignore_index * torch.ones(check_len, dtype=torch.int32).to(topk_tensor.device)  # [-100, ..., -100]
            target[:c] = target_phrase[:c]  # [tok0, tok1, ...tokc, -100, ..., -100]

            # Find sub target matches
            match = (topk_tokens[b, :, :check_len] == target)
            match_idx = torch.where(match.sum(dim=-1) == check_len)[0]  # phrase indices which match sub target

            if len(match_idx):  # at least one match
                match_probs[b] += n_topk_probs[b, match_idx].sum()  # accumulate all matches
            else:  # no matches
                match_probs[b] += n_floor_probs[b]  # assume match is in non-topk tokens with avg floor_prob

    val_probs = torch.clamp(val_probs, 0, 1)  # [batch_size] ensure 0 <= total probability <= 1
    loss_val = - torch.log(val_probs + 1e-40)  # [batch_size] calculate cross entropy loss

    match_probs = torch.clamp(match_probs, 0, 1)  # [batch_size] ensure 0 <= total probability <= 1
    loss = - torch.log(match_probs + 1e-40)  # [batch_size] calculate cross entropy loss

    if reduce:
        if not hasattr(loss_val, reduction) or not hasattr(loss, reduction):
            raise RuntimeError(f'phase_cross_entropy(): Reduction function {reduction} not found.')
        loss_val = getattr(loss_val, reduction)()
        loss = getattr(loss, reduction)()
        if loss.numel() > 1:
            raise ValueError(f'phase_cross_entropy(): Expected reduction to scalar, obtained {loss.shape} instead.')

    return loss_val, loss


def topk_tokens_to_vocab_size(topk_tensor: torch.Tensor, vocab_size_std: int, vocab_size_min: int = 50257) -> torch.Tensor:
    r"""
    Convert topk_tokens first token probabilities into a standard logits tensor shape [batch_size, vocab_size_std].
        Args:
            topk_tensor (:obj:`torch.Tensor`, `required`):
                [batch_size, (topk + 1), max_len] tensor includes topk token probabilities (prob_k) + floor_prob
                in first column with gradients attached, with std_tokens in remaining columns with ignore_index padding.
                Content structure:
                [[[prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., ignore_index?],
                  [prob_k=1_b=0, tok_0_k=1_b=0, tok_1_k=1_b=0, ..., ignore_index?],
                  [...],
                  [prob_floor_b=0, ignore_index, ..., ignore_index]],
                 [[prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., ignore_index?],
                  [prob_k=1_b=1, tok_0_k=1_b=1, tok_1_k=1_b=1, ..., ignore_index?],
                  [...],
                  [prob_floor_b=1, ignore_index, ..., ignore_index]],
                 [...]]
            vocab_size_std (:obj:`int`, `optional`):
                Standard tokenizer vocab_size for forming logits.
            vocab_size_min (:obj:`int`, `optional`):
                Minimum server vocab_size expected, should set to nominal 50257,
                used to prevent the floor_probs from being too large.
        Returns:
            logits (:obj:`torch.Tensor`, `required`):
                [batch_size, vocab_size_std] Standard logits.
    """

    batch_size, topk_p1, max_len = topk_tensor.shape  # [batch_size, (topk + 1), max_len]
    topk = topk_p1 - 1

    topk_tokens = topk_tensor[:, :-1, 1].round().to(torch.int64)  # [batch_size, topk] first tokens
    topk_probs = topk_tensor[:, :-1, 0]  # [batch_size, topk] Probabilities for each phrase in topk
    floor_probs = topk_tensor[:, -1, 0]  # [batch_size] Floor probabilities as mean probability for non-topk tokens

    topk_probs = torch.clamp(topk_probs, 0, 1)  # [batch_size, topk] ensure probabilities within [0, 1]
    floor_probs = torch.clamp(floor_probs, 0, 1)  # [batch_size] ensure floor probabilities within [0, 1]

    # === Ensure total probability is 1 ===
    total_probs = topk_probs.sum(dim=-1) + max(0, vocab_size_min - topk) * floor_probs  # [batch_size] total probs
    n_topk_probs = topk_probs / total_probs[:, None]  # [batch_size, topk] normalized topk_probs

    # === Convert to logits tensor ===
    probs = torch.zeros((batch_size, vocab_size_std))  # [batch_size, vocab_size_std]
    probs.scatter_add_(1, topk_tokens, n_topk_probs)  # accumulate token probabilities onto logits tensor

    return probs  # [batch_size, vocab_size_std]


def check_tokenizer_equivalence(tokenizer_to_check: PreTrainedTokenizerBase,
                                target_tokenizer: PreTrainedTokenizerBase) -> bool:
    r"""
    Is tokenizer_to_check equivalent to target_tokenizer?
        Args:
            tokenizer_to_check (:obj:`PreTrainedTokenizerBase`, `required`):
                Tokenizer to check for equivalence.
            target_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Target tokenizer to check equivalence against.

        Returns:
            result (:obj:`bool`, `required`)
    """
    set_vocab_len(tokenizer_to_check)
    set_vocab_len(target_tokenizer)

    if tokenizer_to_check.vocab_len != target_tokenizer.vocab_len:
        return False

    to_check_vocab = tokenizer_to_check.batch_decode(range(tokenizer_to_check.vocab_len))
    target_vocab = target_tokenizer.batch_decode(range(target_tokenizer.vocab_len))

    return to_check_vocab == target_vocab  # indexed tokenizer vocabularies should match


def prune_tokens(inputs: torch.FloatTensor, prune_len: int = 1, margin: int = 3):
    r"""
    Prune tokens from a batch of sequences randomly by removing prune_len tokens from each sequence,
    leaving the end margin intact.
        Args:
            inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len)`, `required`):
                Tensor inputs to have tokens pruned.
            prune_len (:obj:`int`, `optional`):
                Number of tokens to prune from each validation input sequence.
            margin (:obj:`int`, `optional`):
                Number of tokens at the end of the sequence to leave unpruned.
        Returns:
            pruned_inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_len - prune_len)`, `required`)
    """
    seq_len = len(inputs[0])
    if prune_len <= 0:
        return inputs
    elif seq_len - margin < prune_len:
        prune_len = seq_len - margin
    pruned_inputs = []
    for b in range(len(inputs)):
        rand_index = torch.randperm(seq_len - margin)[:prune_len]
        mask = torch.ones(seq_len, dtype=torch.bool)
        mask[rand_index] = False
        pruned_inputs.append(inputs[b, mask])

    return torch.stack(pruned_inputs)


def pad_offsets(offsets_batch: List[List[tuple]], source_offsets_batch: List[List[List[Any]]],
                pad_offsets_batch: List[List[List[Any]]]) -> List[List[List[Any]]]:
    r"""
    Pads specific tuples in offsets_batch, selected by source_offsets_batch with
    associated paddings in pad_offsets_batch.
    Purpose is typically to add padding to align two tokenization offsets at special tokens.
        Args:
            offsets_batch (:obj:`List[List[tuple]]`, `required`):
                    Batch of full input tokenizer offset mappings to be used for alteration
                    [[(left_0, right_0), (left_1, right_1), ...], ...].
            source_offsets_batch (:obj:`List[List[List[Any]]]`, `required`):
                    Batch of tokenizer offset mappings indicating replacement tuples in offsets_batch
                    [[(left_0, right_0), (left_1, right_1), ...], ...].
            pad_offsets_batch (:obj:`List[List[List[Any]]]`, `required`):
                    Batch of offset paddings associated with each source_offsets_batch replacement tuple
                    [[(left_pad_0, right_pad_0), (left_pad_1, right_pad_1), ...], ...].

        Returns:
            new_offsets_batch (:obj:`List[List[List[Any]]]`, `required`):
                    Batch of padded full input tokenizer offset mappings
                    [[(left_0, right_0), (left_1, right_1), ...], ...].
    """
    new_offsets_batch = []
    batch_len = len(offsets_batch)

    for b in range(batch_len):
        new_offsets = []
        pad = 0

        idx = 0
        for left, right in offsets_batch[b]:  # go through original offsets
            if idx < len(source_offsets_batch[b]):
                source_left, source_right = source_offsets_batch[b][idx]
                if left == source_left and right == source_right:  # matching offset found
                    pad_left, pad_right = pad_offsets_batch[b][idx]
                    new_offsets += [(pad_left + pad, pad_right + pad)]  # replace offsets with padded + accum. pad
                    pad += pad_right - right
                    idx += 1
                    continue
            new_offsets += [(left + pad, right + pad)]  # adjust original offsets w/ accum. pad

        new_offsets_batch += [new_offsets]

    return new_offsets_batch


def find_offsets(string: str, substring: str) -> List[List[int]]:
    r"""
    Finds all the [start, end] offsets of substring in string.
    Assumes there is no overlap of substring, nor recursive overlap.
        Args:
            string (:obj:`str`, `required`):
                Main string to find offsets in.
            substring (:obj:`str`, `required`):
                Substring to search for in string.

        Returns:
            offsets (:obj:`List[List[int]]`, `required`):
                Offsets denoting the [start, end] positions of substring in string.
    """
    offsets = []
    idx = string.find(substring)  # find first instance
    while idx != -1:  # found an instance
        offsets += [[idx, idx + len(substring)]]  # add offsets
        idx = string.find(substring, idx + len(substring))  # find next instance

    return offsets


def replace_at_offsets(string: str, offsets: List[List[Any]]) -> Tuple[str, List[List[int]]]:
    r"""
    Replace indicated [left, right] offset positions with a new substring, by
    deleting [left, right] content and adding [left, left+len(substring)] substring,
    adjusting offsets incrementally.
    Assumes an incremental ordered, non-overlapping list of offsets, constructing
    the new string incrementally and recording new offsets.
        Args:
            string (:obj:`str`, `required`):
                Main string to perform replacements for.
            offsets (:obj:`List[List[Any]]`, `required`):
                Offsets where replacements are made with replacement substring
                [[left_0, right_0, substring_0], ...]

        Returns:
            new_string (:obj:`str`, `required`):
                New string where replacements were made.
            new_offsets (:obj:`List[List[Any]]`, `required`):
                New offsets where replacements are now located
                [[left_0, right_0], [left_1, right_1], ...]
    """
    new_string = ''
    new_offsets = []

    prev = 0
    for left, right, substring in offsets:
        new_string += string[prev:left]  # retain preceding string
        new_left = len(new_string)  # advance index

        new_string += substring  # add new substring
        new_right = len(new_string)

        new_offsets += [[new_left, new_right]]  # add offsets

        prev = right  # advance index

    new_string += string[prev:]

    return new_string, new_offsets


def get_special_token_pairings(from_tokenizer: PreTrainedTokenizerBase,
                               to_tokenizer: PreTrainedTokenizerBase) -> Dict[str, str]:
    r"""
    Determines a prioritized matching of special token texts between two tokenizers.
    Purpose is to produce replacement pairs so special token test is correctly represented for target tokenizer.
        Args:
            from_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                From tokenizer.
            to_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                To tokenizer.

        Returns:
            pairings (:obj:`Dict[str, str]`, `required`):
                Prioritized dictionary of From_special_token_text -> To_special_token_text.
    """
    pairings = {}

    # some tokenizers e.g. GPT2 have the same text signifying BOS and EOS, while in other e.g. XGLM they differ
    # so prioritize EOS token first, since this seems to be the default context separator, e.g. XGLM, GerPT2, GPT2
    if ('eos_token' in from_tokenizer.special_tokens_map) and ('eos_token' in to_tokenizer.special_tokens_map):
        pairings[getattr(from_tokenizer, 'eos_token')] = getattr(to_tokenizer, 'eos_token')

    for special_token in from_tokenizer.special_tokens_map:
        if special_token in to_tokenizer.special_tokens_map:
            if getattr(from_tokenizer, special_token) not in pairings:  # prevent priority overwrite
                pairings[getattr(from_tokenizer, special_token)] = getattr(to_tokenizer, special_token)

    return pairings


def translate_special_token_text(text_batch: List[str], from_tokenizer: PreTrainedTokenizerBase,
                                 to_tokenizer: PreTrainedTokenizerBase) -> Tuple[List[str],
                                                                                 List[List[List[int]]],
                                                                                 List[List[List[int]]],
                                                                                 List[List[List[Any]]]]:
    r"""
    Translates special_token signifier text in from_tokenizer to to_tokenizer special_token text, for
    a given text_batch. Resulting to_text_batch can then be to_tokenized where special_tokens should
    map to its single corresponding token, despite signifier text difference compared to from_tokenizer.
        Args:
            text_batch (:obj:`List[str]`, `required`):
                List of strings to translate special tokens for.
            from_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                From tokenizer.
            to_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                To tokenizer.

        Returns:
            to_text_batch (:obj:`List[str]`, `required`):
                List of strings where special text has been replaced.
            from_offsets_batch (:obj:`List[List[List[int]]]`, `required`):
                Batch of tokenizer offset mappings selecting replacement tuples in from_tokenizer text
                    [[(left_0, right_0), (left_1, right_1), ...], ...].
            to_offsets_batch (:obj:`List[List[List[int]]]`, `required`):
                Batch of tokenizer offset mappings selecting replacement tuples in to_tokenizer text
                    [[(left_0, right_0), (left_1, right_1), ...], ...].
            pad_offsets_batch (:obj:`List[List[List[Any]]]`, `required`):
                Batch of offset paddings associated with each replacement tuple
                    [[(left_pad_0, right_pad_0), (left_pad_1, right_pad_1), ...], ...].
    """
    to_text_batch = []
    from_offsets_batch = []
    to_offsets_batch = []
    pad_offsets_batch = []

    # === Get special-token text replacement pairs ===
    pairings = get_special_token_pairings(from_tokenizer, to_tokenizer)

    for text in text_batch:
        from_offsets = []
        padding_offsets = []
        for token_string in pairings:
            offsets = find_offsets(text, token_string)  # find special-token locations
            from_offsets += [[left, right, pairings[token_string]] for left, right in offsets]

            pad_string = token_string if len(token_string) > len(pairings[token_string]) else pairings[token_string]
            padding_offsets += [[left, right, pad_string] for left, right in offsets]

        from_offsets = sorted(from_offsets)  # incrementally arrange locations
        to_text, to_offsets = replace_at_offsets(text, from_offsets)  # replace special-token text
        pad_text, padding_offsets = replace_at_offsets(text, padding_offsets)  # pad special-token text locations

        to_text_batch += [to_text]
        from_offsets_batch += [[[left, right] for left, right, _ in from_offsets]]
        to_offsets_batch += [to_offsets]
        pad_offsets_batch += [padding_offsets]

    return to_text_batch, from_offsets_batch, to_offsets_batch, pad_offsets_batch


def set_vocab_len(tokenizer: PreTrainedTokenizerBase):
    r"""
    Sets the tokenizer.vocab_len if unset, to store the real vocabulary size according to the vocab or encoder.
        Args:
            tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Tokenizer to set vocab_len for.
        Returns:

    """
    if not hasattr(tokenizer, 'vocab_len'):
        if hasattr(tokenizer, 'vocab'):  # use independent vocab_len when tokenizer.vocab_size != len(tokenizer.vocab)
            tokenizer.vocab_len = len(tokenizer.vocab)
        elif hasattr(tokenizer, 'encoder'):  # tokenizers like facebook/opt-* has encoder=vocab
            tokenizer.vocab_len = len(tokenizer.encoder)
        else:  # revert to vocab_size
            tokenizer.vocab_len = tokenizer.vocab_size


def set_whitespace_preserving(tokenizer: PreTrainedTokenizerBase):
    r"""
    Sets the tokenizer.whitespace_preserving if unset, indicates if tokenizer preserves whitespace like GPT-style,
    or not like BERT-style.
        Args:
            tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Tokenizer to set vocab_len for.
        Returns:

    """
    if not hasattr(tokenizer, 'whitespace_preserving'):
        space_token = tokenizer(' ', add_special_tokens=False)['input_ids']
        space_text = tokenizer.decode(space_token)
        if space_text == ' ':
            tokenizer.whitespace_preserving = True
        else:
            tokenizer.whitespace_preserving = False


def set_std_token_phrases(tokenizer, std_tokenizer):
    r"""
    Sets std_token_phrases which are the tokenizer token strings tokenized with std_tokenizer, so
    the std_tokenizer equivalent of the tokenizer token strings.
    Used for converting model predictions/logits into std_tokenizer representations, for example in TextCausalLMNext.
        Args:
            tokenizer(:obj:`PreTrainedTokenizerBase`, `required`):
                Tokenizer to set std_token_phrases for.
            std_tokenizer(:obj:`PreTrainedTokenizerBase`, `required`):
                Standard bittensor tokenizer to convert to.

        Returns:

    """
    # === Tokenizer phrases to memory ===
    if not hasattr(tokenizer, 'phrases'):
        if tokenizer.whitespace_preserving:
            tokenizer.phrases = tokenizer.batch_decode(range(tokenizer.vocab_len))  # server tokens to strings
        else:
            tokenizer.phrases = [' ' + phrase for phrase in
                                 tokenizer.batch_decode(range(tokenizer.vocab_len))]  # server tokens to strings

    if not hasattr(tokenizer, 'std_token_phrases'):
        # Retokenize phrases to new tokenizer
        tokenizer.std_token_phrases = std_tokenizer(tokenizer.phrases)['input_ids']  # [topk, max_len] convert phrases to tokens sequences


def prep_tokenizer(tokenizer, std_tokenizer=None):
    tokenizer.padding_side = "left"  # Generative default expects most recent token on right-hand side with padding on left. https://github.com/huggingface/transformers/pull/10552
    # tokenizer.add_prefix_space = False
    # tokenizer.add_special_tokens({'bos_token': "[BOS]"}) # A special token representing the beginning of a sentence.
    # tokenizer.add_special_tokens({'eos_token': "[EOS]"}) # A special token representing the end of a sentence.
    # tokenizer.add_special_tokens({'unk_token': "[UNK]"}) # A special token representing an out-of-vocabulary token.
    # tokenizer.add_special_tokens({'sep_token': "[SEP]"}) # A special token separating two different sentences in the same input (used by BERT for instance)
    # tokenizer.add_special_tokens({'pad_token': "[PAD]"}) # A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
    # tokenizer.add_special_tokens({'cls_token': "[CLS]"}) # A special token representing the class of the input (used by BERT for instance).
    # tokenizer.add_special_tokens({'mask_token': "[MASK]"}) # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT).
    # additional_special_tokens = [
    #     "<s>NOTUSED",  # Used by BARThez
    #     "</s>NOTUSED", # Used by BARThez
    #     "<eop>", # Used by MarianMT
    #     "<eod>", # Used by MarianMT
    #     "<formula>", # Used by Transformer XL
    #     "<mask_1>" # Used by Pegasus
    #     "<special0>", # Used by XLM
    #     "<special1>", # Used by XLM
    #     "<special2>", # Used by XLM
    #     "<special3>", # Used by XLM
    #     "<special4>", # Used by XLM
    #     "<special5>", # Used by XLM
    #     "<special6>", # Used by XLM
    #     "<special7>", # Used by XLM
    #     "<special8>", # Used by XLM
    #     "<special9>", # Used by XLM
    # ]
    # tokenizer.additional_special_tokens = additional_special_tokens

    # Define PAD Token = EOS Token (GPT2 generate convention, when PAD Token is None)
    # https://github.com/huggingface/transformers/blob/49c8c67fb815a277405f84dea4a66353e19fb347/tests/models/gpt2/test_modeling_gpt2.py#L532
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    set_vocab_len(tokenizer)
    set_whitespace_preserving(tokenizer)

    if std_tokenizer is not None:
        set_std_token_phrases(tokenizer, std_tokenizer)

    return tokenizer
