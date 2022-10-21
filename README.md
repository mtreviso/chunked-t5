# chunked T5 (cT5)

![Runtime of ct5 and t5 in seconds](./runtime_ct5_vs_t5.png)

A T5 model that uses a new loss where a special end-of-chunk token `</c>` is appended after sentinel tokens. 
The decoder has to predict the full input with masked tokens followed by `</c>`. 
This allows a much faster auto-regressive generation since the decoder can predict multiple tokens in parallel.

For example, for the input `the quick brown fox jumps over the lazy dog`:
```
encoder: the <extra_id_0> fox jumps <extra_id_1> the lazy dog

T5 decoder : <extra_id_0> quick brown <extra_id_1> over <extra_id_2>
cT5 decoder: <extra_id_0> quick brown </c> <extra_id_1> over </c> <extra_id_2>
```

The generation may look like this for T5 and cT5:
```
T5: <extra_id_0>
T5: <extra_id_0> quick
T5: <extra_id_0> quick brown
T5: <extra_id_0> quick brown <extra_id_1>
T5: <extra_id_0> quick brown <extra_id_1> over
T5: <extra_id_0> quick brown <extra_id_1> over <extra_id_2>
T5: <extra_id_0> quick brown <extra_id_1> over <extra_id_2> </s>

cT5: <extra_id_0> <pad> <extra_id_1> <pad> <extra_id_2> </s>
cT5: <extra_id_0> quick <pad> <extra_id_1> over <pad> <extra_id_2> </s>
cT5: <extra_id_0> quick brown <pad> <extra_id_1> over </c> <extra_id_2> </s>
cT5: <extra_id_0> quick brown </c> <extra_id_1> over </c> <extra_id_2> </s>
```

In the original T5, the decoder is called $n_s + 1 + \sum_i |s_i|$ times autoregressively, 
where $n_s$ is the number of sentinel tokens and $s_1,...,s_{n_s}$ are the predicted chunks. 
In contrast, cT5's decoder is called just $max_i |s_i| + 1$ times. 
The generation stops when all sentences were fully translated to complete chunks, i.e., until all `</c>` tokens were generated. 
Alternatively, you can also set `max_chunk_size` to manually force the model to stop after generating a chunk with `max_chunk_size` tokens.
The overhead of calling the decoder with a longer input is less pronounced since this computation can be parallelized in GPUs/TPUs.

## Training details

cT5 models used T5's weights as a starting point, and then it was finetuned on the 
English [wikipedia](https://huggingface.co/datasets/wikipedia) for 3 epochs, 
achieving ~74% validation accuracy (ct5-small and ct5-base).
The training script is in JAX + Flax and can be found in `pretrain_ct5.py`.

Flax checkpoints can be converted to PyTorch via `convert_flax_to_pytorch.py [flax_dirname]`.


## Checkpoints

- ct5-small: https://huggingface.co/mtreviso/ct5-small-en-wiki
- ct5-base: https://huggingface.co/mtreviso/ct5-base-en-wiki
- ct5-large: todo


## Usage

```python
from transformers import AutoTokenizer
from modeling_ct5 import CT5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("mtreviso/ct5-small-en-wiki")
model = CT5ForConditionalGeneration.from_pretrained("mtreviso/ct5-small-en-wiki")
```

For training:

```python
input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> man </c> <extra_id_1> the </c> <extra_id_2>", return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits
```

For generation:

```python
texts = [
    "The <extra_id_0> walks in <extra_id_1> park",
    "UN Chief says there is no way to <extra_id_0> in Syria",
]
input_ids = tokenizer(texts, return_tensors="pt", padding=True).input_ids
generated_ids = model.generate(
    input_ids, 
    use_cache=False,  # important to set to False to avoid caching
    eoc_token_id=tokenizer.vocab['</c>'],  # important to set to the correct end-of-chunk id
    max_chunk_size=5,  # the default is 9999999, which is a large number
)
```

This will produce the following tokens:
```python
>> ['<pad>', '<extra_id_0>', '▁Walking', '▁Trail', '</c>', '<extra_id_1>', '▁the', '</c>', '<extra_id_2>', '</s>']
>> ['<pad>', '<extra_id_0>', '▁treat', '▁Syria', '</c>', '<extra_id_1>', '</s>', '<pad>', '<pad>', '<pad>']
```

You have to pass `use_cache=False` to `generate()` in order to avoid caching during the generation procedure as caching is not available for parallel decoding. 
Currently, parallel decoding is only supported for PyTorch (greedy search, greedy sampling, beam search, beam sampling) and JAX (greedy search and greedy sampling).

**Note on the beam search implementation**: my beam search implementation is slower than optimal, even though it is faster than the vanilla left-to-right beam search.
This is because I use the structures provided by HuggingFace's implementation, namely, BeamScores and BeamHypotheses to store the beam search results for each chunk in the input.
In other words, my implementation computes independent "beams" for each chunk rather than for each input sequence.
It is possible to make it faster by using a custom BeamScores and BeamHypotheses class, but I haven't done that yet. 


## Evaluation

See the notebook `evaluate_ct5.ipynb` for an example of how to evaluate cT5 in terms of accuracy and perplexity.
The notebook `profile.ipynb` shows how to profile the model to get runtimes.

Here is a comparison between cT5-small and T5-small on a subset of the WikiText-103 dataset using deterministic greedy search:

| Model | Exact match ↑ | Fuzzy match ↑ | Perplexity ↓ | Time (seconds) ↓ |
|-------|---------------|----------------------|--------------|-----------------|
| T5-small | 0.11          | 0.60                 | 2.22         | 44.71           |
| cT5-small | 0.09          | 0.58                 | 1.48         | 10.63           |

On this toy dataset, cT5-small has a lower perplexity while being faster than T5-small. However, more experiments are needed for a rigorous evaluation.

If you are interested in applying cT5 to real data, please contact me.


## New eval

| Model | Exact match ↑ | Fuzzy match ↑ |  Word Error Rate ↓ | Perplexity ↓ | Perplexity by GPT-2 ↓ | BERTScore F1 ↑ |  Time (seconds) ↓ |
|-------|---------------|---------------|--------------------|--------------|-----------------------|----------------|-------------------|
| T5-small  | 0.121 | 0.451 | 0.184 | 12.70 | 179.85 | 57.53 | 44.71 |
| cT5-small | 0.095 | 0.580 | 0.169 | 16.96 | 266.87 | 57.07 | 17.31 |
| T5-base   | 0.182 | 0.528 | 0.163 | 15.52 | 168.06 | 57.55 | 77.22 |
| cT5-base  | 0.066 | 0.585 | 0.167 | 18.35 | 274.33 | 56.90 | 24.47 |
