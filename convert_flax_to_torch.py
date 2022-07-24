# export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

import sys


if __name__ == '__main__':

    # dirname = 'ct5-small-en-wiki'
    dirname = sys.argv[1]
    dirname_new = dirname + '-pytorch'
    config_file = dirname + '/config.json'
    checkpoint_file = dirname + '/flax_model.msgpack'
    tokenizer_config_file = dirname + '/tokenizer_config.json'
    tokenizer_file = dirname + '/tokenizer.json'
    special_tokens_map_file = dirname + '/special_tokens_map.json'

    # convert config
    from transformers import T5Config
    config = T5Config.from_json_file(config_file)
    config.save_pretrained(dirname_new)
    print(config)

    # convert tokenizer
    from transformers import T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained(dirname)
    tokenizer.save_pretrained(dirname_new)
    itos = dict(zip(tokenizer.vocab.values(), tokenizer.vocab.keys()))
    print(tokenizer)
    print(len(tokenizer))
    print(tokenizer.vocab['<mask>'], tokenizer.vocab['</c>'])
    print(itos[32100], itos[32101])

    # convert model
    from transformers import T5ForConditionalGeneration
    from transformers.modeling_flax_pytorch_utils import load_flax_checkpoint_in_pytorch_model
    model = T5ForConditionalGeneration(config)
    model = load_flax_checkpoint_in_pytorch_model(model, checkpoint_file)
    model.save_pretrained(dirname_new)
    print(model.shared.weight.mean(), model.shared.weight.std())
