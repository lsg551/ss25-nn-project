"""CrossEncoderBERT is a BERT-based cross-encoder for pointwise and pairwise
ranking tasks.

NOTE: For listewise ranking, BERT has usually not enough context to process all
candidates at once, so a model with larger context is required. A quick empirical
test showed that all MS MARCO candidates in a single sample usually exceed BERT's
512 token limit (between 500 and 1200 tokens). `CrossEncoderLongformer` is an
alternative for listewise ranking. Although it introduced a lot of bias or even
renders any comparison meaningless, it's at least BERT-like.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

MODEL = "bert-base-uncased"
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(MODEL)


class PointwiseRankingHead(nn.Module):
    """Classification head for pointwise ranking.

    For pointwise ranking, the model predicts a relevance score for each
    (query, candidate) pair independently. I.e., the output layer is =1 and
    returns a single logit. After processing all pairs separately, the relevance
    scores can be sorted to get the final ranking (e.g., via a sigmoid mask).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, input_dim // 2)
        # NOTE: ReLU seems to cause gradient issues sometimes (although no proof),
        # so sticking with LeakyReLU or GELU
        self.activation1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(input_dim // 2, output_dim)

    # as input, takes the pooled or raw CLS token output
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        return x


class PairwiseRankingHead(nn.Module):
    """Classification head for pairwise ranking.

    For pairwise ranking, the model predicts a relevance score for each
    (query, candidate A, candidate B) triplet. The objective is to …

    There are two possible implementations:

    1. Two logits output

    2. Two forward passes.
    """

    ...
    # TODO: implement


# TODO: add a property to let the user decide which model to use (pointwise,pairwise)


class CrossEncoderBERT(nn.Module):
    """CrossEncoderBERT is a simple BERT-based cross-encoder for passage reranking.

    The model is composed of a pre-trained BERT model with a simple regression
    head. Each single forward pass returns the raw logits of the regression head.
    This can be interpreted as the relevance score of the candidate given the query.

    Example:
    >>> from models import CrossEncoderBERT
    >>> query, candidate = "What is AI?", "AI is the field of study that ..."
    >>> model = CrossEncoderBERT()
    >>> logits = model(*model.tokenize(query, candidate))

    For pairwise passage ranking, all pairs can be sorted by the raw logits
    once all candidates have been scored.
    """

    def __init__(
        self,
        model_name: str = MODEL,
        *,
        enable_gradient_checkpointing: bool = False,
    ) -> None:
        """Create a new CrossEncoderBERT model.

        Args:
            model_name (str, optional): Hugging Face model identifier.
                Defaults to MODEL.
            enable_gradient_checkpointing (bool, optional): Enable gradient
                checkpointing to save activation memory at cost of extra compute.
                Defaults to False.
        """
        super().__init__()
        self.bert: BertModel = BertModel.from_pretrained(model_name)

        if enable_gradient_checkpointing:
            if hasattr(self.bert.config, "use_cache"):
                # disable use_cache if gradient checkpointing is enabled
                self.bert.config.use_cache = False
            self.bert.gradient_checkpointing_enable()

        self.classifier = PointwiseRankingHead(
            input_dim=self.bert.config.hidden_size,
            output_dim=1,  # 1 = raw logit
        )

    def forward(
        self,
        input_ids: torch.Tensor,  # token IDs in the vocab
        attention_mask: torch.Tensor,  # mask for padding tokens
        token_type_ids: torch.Tensor,  # for sentence pairs
    ):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # NOTE: HF's implementation of BertModel adds an extra layer on top of
        # the CLS token output with the same shape. It is (/was) recommended to
        # use `pooler_output` instead of `last_hidden_state[:, 0, :]`. Howevser,
        # the raw [CLS] hidden state may perform better in some cases.
        # Only BERT implements this, newer models like RoBERTa don't have it.

        # cls_output = out.last_hidden_state[:, 0] # [:, 0, :] == [:, 0]
        cls_output = out.pooler_output  # shape

        logits = self.classifier(cls_output)

        return logits

    def tokenize(
        self,
        queries: str | list[str],
        candidates: str | list[str],
        *,
        padding: str = "longest",
        max_length: int = 512,
    ):
        """Tokenize one or many (query, candidate) pairs.

        Args:
            query: A single query string or a list of queries.
            candidate: A single candidate string or a list of candidates.
            padding: Padding strategy. See
                https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__
                for details. Defaults to "longest" (dynamic padding).
            max_length: Optional max length for truncation/padding. If None,
                uses the model's default max length (usually 512 for BERT).

        Returns:
            BatchEncoding: A dictionary-like object with the following
                fields: input_ids, attention_mask, token_type_ids etc.
        """
        return tokenizer(
            queries,
            candidates,
            truncation=True,
            padding=padding,
            max_length=max_length,
            return_tensors="pt",
        )

    def rank(
        self, query: str, candidates: list[str], *, device: str | torch.device
    ) -> list[tuple[str, float]]:
        self.eval()
        inputs = self.tokenize([query] * len(candidates), list(candidates)).to(device)
        with torch.no_grad():
            logits = self(
                input_ids=inputs["input_ids"],  # type: ignore
                attention_mask=inputs["attention_mask"],  # type: ignore
                token_type_ids=inputs["token_type_ids"],  # type: ignore
            )
        scores = torch.sigmoid(logits).squeeze(-1).tolist()
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
