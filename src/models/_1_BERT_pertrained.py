import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

MODEL = "bert-base-uncased"

tokenizer: BertTokenizer = BertTokenizer.from_pretrained(MODEL)

# NOTE: we _could_ also use BertForSequenceClassification, which adds a
# classification head on top of BERT as well as loss computation ...
#


class ClassificationHead(nn.Module): ...


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

    def __init__(self, model_name=MODEL):
        super(CrossEncoderBERT, self).__init__()
        self.bert: BertModel = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.1)
        # self.activation = nn.ReLU()

        # Ba et al.: Layer Normalization (2016), cf. https://arxiv.org/pdf/1607.06450
        # self.layernorm = nn.LayerNorm(self.bert.config.hidden_size)

        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

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
        cls_output = out.pooler_output

        logits = self.classifier(self.dropout(cls_output))

        return logits

    @staticmethod
    def tokenize(query: str | list[str], candidate: str | list[str]):
        return tokenizer(
            query,
            candidate,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
