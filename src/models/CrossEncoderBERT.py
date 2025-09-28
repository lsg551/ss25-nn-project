import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

MODEL = "bert-base-uncased"

tokenizer: BertTokenizer = BertTokenizer.from_pretrained(MODEL)

# NOTE: we _could_ also use BertForSequenceClassification, which adds a
# classification head on top of BERT as well as loss computation ...
#


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 2 * input_dim)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(2 * input_dim, output_dim)

    # as input, takes the pooled or raw CLS token output
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        return x


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
        dropout: float = 0.1,
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

        self.classifier = ClassificationHead(
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
