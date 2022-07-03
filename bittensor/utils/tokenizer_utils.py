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

from typing import List, Dict, Tuple, Any
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

    phrases = tokenizer.batch_decode(range(len(tokenizer.vocab)))  # list of variable len strings (one per token)

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
    translation_map = {'lengths': {}}

    phrases = from_tokenizer.batch_decode(range(len(from_tokenizer.vocab)))  # tokens to strings

    to_tokens = to_tokenizer(phrases)['input_ids']  # convert single token from-phrases to to-tokenization
    to_tokens_lens = [len(p) for p in to_tokens]
    unique_lens = set(to_tokens_lens)
    max_len = max(unique_lens)
    counts = torch.zeros((max_len, len(to_tokenizer.vocab)), dtype=torch.long)

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
                    [sequence_len] A sequence of tokens produced by the source tokenizer.
                tokens_std (:obj:`torch.LongTensor`, `required`):
                    [std_sequence_len] A sequence of tokens produced by the standard tokenizer.
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
    # === Check tokenizer equivalence / Skip if equivalent ===
    if skip_equivalent and check_tokenizer_equivalence(tokenizer, std_tokenizer):
        probs = torch.softmax(logits, dim=2).to('cpu')
        return probs

    # === Get shape sizes ===
    batch_size, sequence_len, vocab_size = logits.shape
    std_sequence_len = max([len(seq) for seq in offset_mapping_std])
    std_vocab_size = len(std_tokenizer.vocab)

    # === Convert logits to probabilities ===
    probs = torch.softmax(logits, dim=2).to('cpu')  # [batch_size, sequence_len, vocab_size]

    if vocab_size < len(tokenizer.vocab):  # fixes bug when model logits output is not full width
        padded_probs = torch.zeros((batch_size, sequence_len, len(tokenizer.vocab)))
        padded_probs[..., :vocab_size] = probs
        probs = padded_probs

    # === Translate to probabilities over standard tokenizer ===
    probs_std = torch.zeros(batch_size, std_sequence_len, std_vocab_size)
    for b in range(batch_size):
        translate_tokenizer_probs(probs[b], probs_std[b], offset_mapping[b], offset_mapping_std[b],
                                  tokenizer, std_tokenizer,
                                  split_map_cache, to_translation_map, from_translation_map,
                                  tokens[b], tokens_std[b])

    # === Correct excess probability mass (haircut) ===
    probs_std_sum = probs_std.sum(dim=-1)  # [batch_size, std_sequence_len]
    over = (probs_std_sum > 1)
    probs_std[over] /= probs_std_sum[over][:, None]

    # === Correct deficient probability mass (raise) ===
    probs_std_sum = probs_std.sum(dim=-1)  # [batch_size, std_sequence_len]
    under = (probs_std_sum < 1)
    probs_std[under] += ((1 - probs_std_sum[under]) / probs_std[under].shape[-1])[:, None]  # raise noise floor so sum 1

    return probs_std  # [batch_size, std_sequence_len, std_vocab_size]


def topk_token_phrases(logits: torch.Tensor, tokenizer: PreTrainedTokenizerBase, to_tokenizer: PreTrainedTokenizerBase,
                       topk: int, ignore_index: int = -100) -> Tuple[torch.Tensor,
                                                                     torch.Tensor,
                                                                     torch.Tensor,
                                                                     torch.Tensor]:
    r"""
    Select topk tokenizer logits and retokenize with to_tokenizer, then compact new token phrases and probabilities
    into 1-D tensor [ > batch_size * 2 * topk + 1] prob + at least 1 token per phrase + floor_prob.
    The floor probability is the mean probability of token phrases not captured in topk, required since
    the tokenizer vocab_size may not be known to the receiver.
        Args:
            logits (:obj:`torch.Tensor`, `required`):
                [batch_size, vocab_size] Input source logits for last token over a source tokenizer vocabulary.
            tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Source tokenizer (usually server tokenizer)
            to_tokenizer (:obj:`PreTrainedTokenizerBase`, `required`):
                Target tokenizer (usually standard validator tokenizer)
            topk (:obj:`int`, `required`):
                Amount of top phrases to expect (to check for mismatch)
            ignore_index (:obj:`int`, `optional`):
                Padding value to use for unfilled token positions in a shorter token phrase.

        Returns:
            compact_topk (:obj:`torch.Tensor`, `required`):
                [sum_b(sum_k(len(phrase_k) + 1)_b)] Compacted 1-D tensor > batch_size * 2 * topk + 1,
                since 2 * topk + 1: topk x [probability, token sequence (at least one token)] +
                floor probability (rest).
                Content structure:
                [prob_k=0_b=0, tok_0_k=0_b=0, tok_1_k=0_b=0, ..., prob_k=1_b=0, tok_0_k=1_b=0, ..., prob_floor_b=0,
                 prob_k=0_b=1, tok_0_k=0_b=1, tok_1_k=0_b=1, ..., prob_k=1_b=1, tok_0_k=1_b=1, ..., prob_floor_b=1,
                 ...]
            topk_tokens (:obj:`torch.Tensor`, `required`):
                [batch_size, topk, max_len] Phrase tokens with ignore_index token for padding.
            topk_probs (:obj:`torch.Tensor`, `required`):
                [batch_size, topk] Probabilities for each phrase in topk.
            floor_probs (:obj:`torch.Tensor`, `required`):
                [batch_size] Floor probabilities as mean probability for non-topk tokens.
    """
    # Get shape sizes
    batch_size, vocab_size = logits.shape  # [batch_size, vocab_size] only last token prediction

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1).to('cpu')  # [batch_size, vocab_size]

    # TopK phrase selection
    topk_probs, topk_indices = torch.topk(probs, topk)  # topk probs and indices: [batch_size, topk]

    # === Calculate floor probability ===
    topk_pmass = topk_probs.sum(dim=-1)  # [batch_size] topk probability mass
    remainder_pmass = torch.clamp(1 - topk_pmass, 1e-40, 1)  # [batch_size] remainder probability mass
    floor_probs = remainder_pmass / (vocab_size - topk)  # [batch_size]divide remainder

    # === Tokenizer phrases to memory ===
    if not hasattr(tokenizer, 'phrases'):
        tokenizer.phrases = tokenizer.batch_decode(range(len(tokenizer.vocab)))  # server tokens to strings

    tensors = []
    tokens_batch = []

    # === Construct topk batch ===
    for b in range(batch_size):
        # Select topk phrases
        phrases = [tokenizer.phrases[i] for i in topk_indices[b]]  # str[topk]

        # Retokenize phrases to new tokenizer
        to_tokens = to_tokenizer(phrases)['input_ids']  # [topk, max_len] convert phrases to tokens sequences

        token_phrases = []
        # === Reassemble topk info ===
        for i in range(topk):
            phrase_tensor = torch.tensor(to_tokens[i], requires_grad=True, dtype=torch.float)
            token_phrases += [phrase_tensor]
            tensors += [topk_probs[b, i], phrase_tensor + 2]  # increment 2 to reserve [0, 1] for probs

        tensors += [floor_probs[b]]
        tokens_batch += [token_phrases]

    compact_topk = torch.hstack(tensors).to(torch.float32)  # [sum_b(sum_k(len(phrase_k) + 1)_b)]

    # === Tensorize topk token selection ===
    max_len = max([max([len(phrase) for phrase in phrases]) for phrases in tokens_batch])  # max_len
    topk_tokens = ignore_index * torch.ones((len(tokens_batch), topk, max_len))  # [batch_size, topk, max_len]

    for b, phrases in enumerate(tokens_batch):
        for k, phrase in enumerate(phrases):
            topk_tokens[b, k, :len(phrase)] = phrase

    topk_tokens = topk_tokens.to(int)

    return compact_topk, topk_tokens, topk_probs, floor_probs


def unravel_topk_token_phrases(input_tensor: torch.Tensor, topk: int, ignore_index: int = -100) -> Tuple[torch.Tensor,
                                                                                                         torch.Tensor,
                                                                                                         torch.Tensor]:
    r"""
    Unravel topk token phrases input_tensor from 1-D to [batch_size, topk, max_len].
        Args:
            input_tensor (:obj:`torch.Tensor`, `required`):
                [sum_b(sum_k(len(phrase_k) + 1)_b)] Compacted 1-D tensor > batch_size * 2 * topk + 1,
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
            topk_tokens (:obj:`torch.Tensor`, `required`):
                [batch_size, topk, max_len] Phrase tokens with ignore_index token for padding.
            topk_probs (:obj:`torch.Tensor`, `required`):
                [batch_size, topk] Probabilities for each phrase in topk.
            floor_probs (:obj:`torch.Tensor`, `required`):
                [batch_size] Floor probabilities as mean probability for non-topk tokens.
    """

    # Find probability markers
    prob_idx = torch.where(input_tensor <= 1)[0]  # 0 <= prob <= 1

    # Decrement token ids
    input_tensor[input_tensor >= 2] -= 2  # decrement token id to original value

    probs = []
    phrases = []
    tokens_batch = []
    probs_batch = []
    floor_probs = []

    # === Extract phrase info ===
    prev_idx = prob_idx[0] + 0
    prob = input_tensor[prev_idx] + 0
    for idx in prob_idx[1:]:
        if prev_idx + 1 == idx:  # encounter floor probability, create new batch_item
            floor_probs += [prob]
            tokens_batch += [phrases]
            probs_batch += [torch.stack(probs)]
            phrases = []
            probs = []
        else:
            probs += [prob]
            phrases += [input_tensor[prev_idx + 1:idx]]  # torch.tensor([prob, tok0, tok1, ...tokn])

        prev_idx = idx + 0
        prob = input_tensor[prev_idx] + 0

    tokens_batch += [phrases]

    probs_batch += [torch.stack(probs)]
    topk_probs = torch.vstack(probs_batch)  # [batch_size, topk] phrase probability

    floor_probs += [prob]  # last floor probability
    floor_probs = torch.stack(floor_probs)  # [batch_size] floor probabilities as mean probability for non-topk tokens

    # === Check batch items for same topk ===
    if 0 < sum([not len(phrases) == topk for phrases in tokens_batch]):
        raise ValueError('reshape_topk_token_phrases(): topk lengths unmatched.')

    # === Tensorize topk token selection ===
    max_len = max([max([len(phrase) for phrase in phrases]) for phrases in tokens_batch])  # max_len
    topk_tokens = ignore_index * torch.ones((len(tokens_batch), topk, max_len))  # [batch_size, topk, max_len]

    for b, phrases in enumerate(tokens_batch):
        for k, phrase in enumerate(phrases):
            topk_tokens[b, k, :len(phrase)] = phrase

    topk_tokens = topk_tokens.to(int)  # [batch_size, topk, max_len]

    return topk_tokens, topk_probs, floor_probs


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
    if len(tokenizer_to_check.vocab) != len(target_tokenizer.vocab):
        return False

    to_check_vocab = tokenizer_to_check.batch_decode(range(len(tokenizer_to_check.vocab)))
    target_vocab = target_tokenizer.batch_decode(range(len(target_tokenizer.vocab)))

    return to_check_vocab == target_vocab  # indexed tokenizer vocabularies should match


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
