# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0. See the original
# LLaMA-Factory file for the full notice.
#
# Differences from upstream:
#   - Adds a `ComputePerplexity` metric (with a paired
#     `eval_logit_processor_logprobs`) so that the MIA paper's reported PPL
#     can be reproduced. Use it on a teacher-forced eval pass
#     (`predict_with_generate: false`).

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional
import warnings

import numpy as np
import torch
from transformers.utils import is_jieba_available, is_nltk_available

from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_rouge_available

from pycocoevalcap.bleu.bleu import Bleu  # noqa: F401  (kept for parity with upstream)
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from nltk.translate.bleu_score import corpus_bleu  # noqa: F401


if is_rouge_available():
    from rouge_chinese import Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    """Default upstream behaviour: collapse (B, T, V) logits to argmax tokens."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


def eval_logit_processor_logprobs(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    """Return log-softmax logits, used by ComputePerplexity instead of argmax.

    Shape preserved as (B, T, V). For long contexts this can be expensive in
    memory — use a small per_device_eval_batch_size.
    """
    if isinstance(logits, (list, tuple)):
        logits = logits[0] if logits[0].dim() == 3 else logits[1]
    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")
    return torch.log_softmax(logits.float(), dim=-1)


@dataclass
class ComputeAccuracy:
    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Wraps the tokenizer into BLEU / ROUGE / Distinct / METEOR / CIDEr metrics.

    Used by CustomSeq2SeqTrainer on a generation pass (predict_with_generate=true).
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {
            "rouge-1": [], "rouge-2": [], "rouge-l": [],
            "bleu-1": [], "bleu-2": [], "bleu-3": [], "bleu-4": [],
            "dist-1": [], "dist-2": [], "dist-3": [],
            "Meteor": [], "Cider": [],
        }
        return result

    def __post_init__(self):
        self._dump()

    def calc_distinct_k(self, ngram, hyps):
        d = {}
        tot = 0
        for sen in hyps:
            for i in range(0, len(sen) - ngram):
                key = tuple(sen[i:i + ngram])
                d[key] = 1
                tot += 1
        if tot > 0:
            dist = len(d) / tot
        else:
            warnings.warn("the distinct is invalid")
            dist = 0.0
        return dist

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_dict, label_dict = {}, {}
        for idx, (pred, label) in enumerate(zip(decoded_preds, decoded_labels)):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_1 = sentence_bleu([list(label)], list(pred),
                                   smoothing_function=SmoothingFunction().method3,
                                   weights=(1, 0, 0, 0))
            bleu_2 = sentence_bleu([list(label)], list(pred),
                                   smoothing_function=SmoothingFunction().method3,
                                   weights=(0.5, 0.5, 0, 0))
            bleu_3 = sentence_bleu([list(label)], list(pred),
                                   smoothing_function=SmoothingFunction().method3,
                                   weights=(0.33, 0.33, 0.33, 0))
            bleu_4 = sentence_bleu([list(label)], list(pred),
                                   smoothing_function=SmoothingFunction().method3,
                                   weights=(0.25, 0.25, 0.25, 0.25))
            self.score_dict["bleu-1"].append(round(bleu_1 * 100, 4))
            self.score_dict["bleu-2"].append(round(bleu_2 * 100, 4))
            self.score_dict["bleu-3"].append(round(bleu_3 * 100, 4))
            self.score_dict["bleu-4"].append(round(bleu_4 * 100, 4))

            key = str(idx + 1)
            pred_dict[key] = [pred]
            label_dict[key] = [label]

        try:
            m = Meteor()
            score, _ = m.compute_score(gts=label_dict, res=pred_dict)
            self.score_dict["Meteor"].append(score * 100)
        except Exception:
            self.score_dict["Meteor"].append(0)

        try:
            c = Cider()
            score, _ = c.compute_score(gts=label_dict, res=pred_dict)
            self.score_dict["Cider"].append(score * 100)
        except Exception:
            self.score_dict["Cider"].append(0)

        dist_1 = self.calc_distinct_k(ngram=1, hyps=decoded_preds)
        dist_2 = self.calc_distinct_k(ngram=2, hyps=decoded_preds)
        dist_3 = self.calc_distinct_k(ngram=3, hyps=decoded_preds)
        self.score_dict["dist-1"].append(round(dist_1 * 100, 4))
        self.score_dict["dist-2"].append(round(dist_2 * 100, 4))
        self.score_dict["dist-3"].append(round(dist_3 * 100, 4))

        if compute_result:
            return self._dump()


@dataclass
class ComputePerplexity:
    r"""Token-level perplexity on the response span.

    Pair with `eval_logit_processor_logprobs` (NOT the default argmax processor)
    and run a teacher-forced eval pass (`predict_with_generate: false`). The
    `EvalPrediction.predictions` then carries log-softmax logits of shape
    (B, T, V), which we shift one position and gather at the gold-label index.
    """

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict") and self.score_dict["nll"]:
            mean_nll = float(np.mean(self.score_dict["nll"]))
            result = {"ppl": float(np.exp(mean_nll)), "nll": mean_nll}
        elif hasattr(self, "score_dict"):
            result = {"ppl": float("nan"), "nll": float("nan")}
        self.score_dict = {"nll": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        logprobs = numpify(eval_preds.predictions)  # (B, T, V)
        labels = numpify(eval_preds.label_ids)      # (B, T)
        if logprobs.ndim != 3:
            raise ValueError("ComputePerplexity expects log-softmax logits; "
                             "use eval_logit_processor_logprobs.")

        # Standard causal-LM shift: pred[i] is the prediction *for* label[i+1].
        logprobs = logprobs[:, :-1, :]
        labels = labels[:, 1:]
        mask = labels != IGNORE_INDEX
        # take_along_axis can't ingest IGNORE_INDEX (-100) — substitute 0 then mask.
        safe = np.where(mask, labels, 0).astype(np.int64)
        gathered = np.take_along_axis(logprobs, safe[..., None], axis=-1).squeeze(-1)
        nll = -gathered[mask]
        if nll.size > 0:
            self.score_dict["nll"].extend(nll.tolist())

        if compute_result:
            return self._dump()
