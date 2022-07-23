# chunked T5 (cT5)

A T5 model that uses a new loss where a special end-of-chunk token `</c>` is appended after sentinel tokens. 
The decoder has to predict the full input with masked tokens followed by `</c>`. 
This allows a much faster auto-regressive generation since the decoder can predict multiple tokens in parallel.

For example:
```
input: the quick brown fox jumps over the lazy dog

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

cT5: <extra_id_0> <pad> <extra_id_1> <pad> <extra_id_2>
cT5: <extra_id_0> quick <pad> <extra_id_1> over <pad> <extra_id_2>
cT5: <extra_id_0> quick brown <pad> <extra_id_1> over </c> <extra_id_2>
cT5: <extra_id_0> quick brown </c> <extra_id_1> over </c> <extra_id_2>
```

In the original T5, the decoder is called $n_s + 1 + \sum_i |s_i|$ times autoregressively, 
where $s_1,...,s_{n_s}$ are the predicted spans. 
In contrast, cT5's decoder is called only $max_i |s_i| + 1$ times. 
The generation stops when all sentences were fully translated to complete chunks, i.e., until all `</c>` tokens appear. 
Alternatively, you can also set `max_chunk_size` to manually force the model to stop after generating a chunk with `max_chunk_size` tokens.
The overhead of calling the decoder with a longer input is less pronounced since this computation can be parallelized in GPUs/TPUs.

## Huggingface Checkpoints

- ct5-small: https://huggingface.co/mtreviso/ct5-small-en-wiki


## Training details

cT5-small used T5's weights as a starting point, and then it was finetuned on the English [wikitext-103](https://huggingface.co/datasets/wikitext) dataset for 3 epochs, achieving ~74% validation accuracy.
The training script is in JAX + Flax and can be found in `pretrain_ct5.py`.

Flax checkpoints can be converted to PyTorch via `convert_flax_to_pytorch.py [flax_dirname]`.
 

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
labels = tokenizer("<extra_id_0> quick brown </c> <extra_id_1> over </c> <extra_id_2>", return_tensors="pt").input_ids
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
>>> ['<pad>', '<extra_id_0>', '▁Walking', '▁Trail', '</c>', '<extra_id_1>', '▁the', '</c>', '<extra_id_2>', '</s>']
>>> ['<pad>', '<extra_id_0>', '▁treat', '▁Syria', '</c>', '<extra_id_1>', '</s>', '<pad>', '<pad>', '<pad>']
```

**Note** that you have to pass `use_cache=False` to `generate()` in order to avoid caching during the generation procedure (not available for parallel decoding). 
Currently, parallel decoding is only supported for PyTorch (greedy search, greedy sampling, beam search, beam sampling) and JAX (greedy search and greedy sampling).


## Evaluation

See the notebook `evaluate_ct5.ipynb` for an example of how to evaluate cT5 in terms of accuracy and running speed.
