# Docs

This document provides a little documentation and supplementary information about
the project, configuration, architecture, etc. as well as propaedeutic material.

## Objectives

- [x] Add support for the MS MARCO dataset
- [x] Add support for the mMARCO dataset
- [x] Implement a pointwise ranking model, based on BERT
    - Successfully implemented (see [`experiments/pointwise.ipynb`](./experiments/pointwise.ipynb)
      AND [`experiments/pointwise_MS_MARCO.ipynb`](./experiments/pointwise_MS_MARCO.ipynb)),
      but the training did not reach satisfying results. The model did not
      converge below < 0.2 loss. I tried many approaches in many attempts,
      without overwhelming success. Learned a lot though.
- [ ] Implement a pairwise ranking model, based on BERT
    - I did not manage to get this implemented in time and faced some theoretical
      issues I could not solve / decide for one way or another.
- [x] Implement a listwise ranking model, based on ~~BERT~~~
    - At least on MS MARCO, the candidates are often longer than BERT's context
      of 512 tokens. I used Allen AI's Longformer, a RoBERTa-like model.
    - <font color=tomato>NOTE</font>: the implementation is so slow (likely
      because of a performance bottleneck somehwere)  that it became infeasible
      to train it on the full dataset or anywhere near a useable state. The
      notebook [`experiments/listwise.ipynb`](./experiments/listwise.ipynb) is
      fully functional though.
    - I tried to dug deeper into PyTorch performance profiling and optimization,
      but much of that was far beyond my current knowledge and scope of this project.
    - I also faced some loss optimization issues, and after some reading, the
      rabbit hole of listwise ranking seemed too deep to me at this point. Some
      of my struggle is documented in the notebook
      [`experiments/listwise.ipynb`](./experiments/listwise.ipynb).
- [x] Add ranking metrics for inter-model comparison on MS MARCO
    - [x] Implement MRR
    - [ ] Implement nDCG
    - [x] Implement Recall@K 
    - [ ] Use them for per-model evaluation
        - partially for pointwise (see
          [`experiments/pointwise_MS_MARCO.ipynb`](./experiments/pointwise_MS_MARCO.ipynb)) 
    - [ ] Perform inter-model comparison
- [ ] BM25 baseline
    - In the end, I omitted this idea, because I ran into issues with the
      MS MARCO corpus regarding truncation / padding of passages
      (see [`experiments/listwise.ipynb`](./experiments/listwise.ipynb)).
- [ ] (optional) read about reproducibility with torch and try to apply it
- [ ] (optional) try something like wantb or DVC for experiment tracking


## Ranking Types

Different things can be ranked, commonly:
- entire documents (e.g., web pages, articles, etc.)
- passages (e.g., paragraphs, sections, document chunks, etc.)

And these things can be put into order in various different fashions:

<table>
<tr>
    <th>#</th>
    <th><i>pairwise</i></th>
    <th><i>pointwise</i></th>
    <th><i>listwise</i></th>
</tr>
<tr>
    <th>Input</th>
    <td>one query-candidate pair</td>
    <td>one query, two candidates</td>
    <td>one query, <i>all</i> candidates</td>
</tr>
<tr>
    <th>Output</th>
    <td>relevance score ({1,0} or logits)</td>
    <td>two relevance scores, one for each candidate</td>
    <td>the entire ranking itself</td>
</tr>
<tr>
    <th>Learning Objective</th>
    <td>predicting (relevance) scores</td>
    <td>kind of differentiation: which candidate scores higher</td>
    <td>generating rankings directly / optimizing ranking metrics directly</td>
</tr>
<tr>
    <th>Loss</th>
    <td>often MSE (regression) or CEL (classification)</td>
    <td>hinge loss or pairwise logistic loss</td>
    <td>
        dependent on specific metrics for ranking
        (e.g. nDCG, ListMLE, LambdaLoss, …)
    </td>
</tr>
<tr>
    <th>Accuracy</th>
    <td>worse than the others, but simple and cheap to implement</td>
    <td>somewhere inbetween</td>
    <td>usually outperforms all others</td>
</tr>
</table>

The Wikipedia article on [*Learning to Rank*]((https://en.wikipedia.org/wiki/Learning_to_rank))
has a good overview and a list with historical and a few recent ranking models.

In recent years, off-the-shelf LLMs have been used with zero- or few-shot
"learning" to create cheap ranking models. Weights of smaller language models
also have been used as base models to the cross-encoder architecture.

## Baselines

Usually, if the goal is to improve SOTA models, the best SOTA rerankers would
be used as baselines. Or, if the goal is cost-efficiency or high-throughput,
SOTA models from these categories would be used as baselines.

Often BM25 is used as a baseline, because it's simple, fast, and surprisingly
effective for many IR tasks.

## Dataset

There are a few datasets available for training cross-encoders on Hugging Face.
Search with multiple keywords like *cross-encoder*, *ranking*, *reranking*,
*reranker*, etc. Many more can be repurposed under certain conditions, e.g., if
hard negative mining is feasible.

<table>
<tr>
  <th>Name</th>
  <th>Description</th>
  <th># Rows</th>
</tr>
<tr>
    <td>
        <a href="https://huggingface.co/datasets/microsoft/ms_marco">
            <code>microsoft/ms_marco</code>
        </a>
    </td>
    <td>
        Popular general purpose dataset for many IR tasks. Introduced at NIPS
        2016 by <a href="https://arxiv.org/pdf/1611.09268">Bajaj et al. (2016)</a>.
        It's based on real user queries from Bing Search with (few) real human
        answers, (more) human-supervised generated answers and (mostly) retrieved
        passages from Bing. Many other datasets are based on / derived from this one.
    </td>
    <td>> 1.1 million</td>
</tr>
<tr>
    <td>
        <a href="https://huggingface.co/datasets/unicamp-dl/mmarco">
            <code>unicamp-dl/mmarco</code>
        </a>
    </td>
    <td>
        <a href="https://arxiv.org/pdf/2108.13897">Bonifacio et al. (2021)</a>
        proposed this. Based on MS MARCO. It's multilingual, presplitted, and
        specifically for ranking. It does not distinguish between hard and soft
        negatives though. Negatives seem to be randomly sampled hard negatives.
    </td>
    <td>
        ?
    </td>
</tr>
<tr>
    <td>
        <a href="https://huggingface.co/datasets/hotchpotch/mmarco-hard-negatives-reranker-score">
            <code>hotchpotch/mmarco-hard-negatives-reranker-score</code>
        </a>
    </td>
    <td>
        …
    </td>
    <td>> 7 million</td>
</tr>
<tr>
    <td>TREC CAR</td>
    <td>
        Dataset for passage reranking introduced by
        <a href="https://trec.nist.gov/pubs/trec26/papers/Overview-CAR.pdf">Dietz et al. (2017)</a>;
        prior to its release, many IR datasets were
        focused on QA or document ranking, but not passage ranking.
        <br/>
        The original url is trec-car.cs.unh.edu, but seems to be no longer
        reachable or not cached by a CDN. There appears to be a mirror though:
        <a href="https://ir-datasets.com/car.html">ir-datasets.com/car.html</a>.
    </td>
    <td>?</td>
</tr>
<tr>
    <td>
        <a href="https://huggingface.co/datasets/samheym/ger-dpr-collection-crossencoder">
            <code>samheym/ger-dpr-collection-crossencoder</code>
        </a>
    </td>
    <td>
        German DPR (<i>dense passage retrieval</i>); massive and open-source,
        but one has to agree to the license before it can be downloaded.
        Unfortunately, it seems like the authors abandoned the HF project, at
        least my request is pending for months without any response.
    </td>
    <td>> 81 million</td>
</tr>
<tr>
    <td>
        <a href="https://huggingface.co/datasets/deepset/germandpr">
            <code>deepset/germandpr</code>
        </a>
    </td>
    <td>
        Based on GermanQuAD with hard negatives minded from German Wikipedia.
    </td>
    <td>
        > 10k
    </td>
</tr>
<tr>
    <td>
        <a href="https://huggingface.co/datasets/mteb/mind_small_reranking">
            <code>mteb/mind_small_reranking</code>
        </a>
    </td>
    <td>
        From the Massive Text Embedding Benchmark (MTEB).
        Huge, but only query-candidate pairs with a binary relevance score.
    </td>
    <td>> 216 million</td>
</tr>
<tr>
    <td><a href="https://huggingface.co/datasets/sentence-transformers/gooaq"><code>sentence-transformers/gooaq</code></a></td>
    <td>Google annotated question-answer pairs. Must be refitted.</td>
    <td>> 3 million pairs</td>
</tr>
<tr>
    <td>
        <a href="https://huggingface.co/datasets/sentence-transformers/msmarco-scores-ms-marco-MiniLM-L6-v2">
            <code>sentence-transformers/msmarco-scores-ms-marco-MiniLM-L6-v2</code>
        </a>
    </td>
    <td>
        MS MARCO + unprocessed query-candidate pairs with logit scores
        processed from the cross-encoder in the name.
    </td>
    <td>
        > 240 million
    </td>
</tr>
</table>

There are many more notable datasets, although many of them were created prior
to standardized sharing platforms like Hugging Face and data formats like
Parquet (or at least before these became mainstream within the research community).

## Loss Function & Metrics

The loss function has to be chosen according to the dataset and task. The
datasets listed above have different formats, but SentenceTransformers provides
a good [overview of what loss functions to use for which task and dataset](https://sbert.net/docs/sentence_transformer/loss_overview.html).

Further more, a few metrics can be used to evaluate the model's "real world" /
downstream task performance:
- various @K metrics, for example: nDCG@K, MRR@K, etc.


## Models



- BERT small, comparable (because many other papers still use it).
  However, BERT can only process up to 512 input tokens at once.
  This is not ideal for listwise ranking with a lot of candidates or
  long candidates.
- Longformer (Allen AI), a RoBERTa-like model that can process longer
  inputs (4096).


