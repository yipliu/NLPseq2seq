1. Packed Padded Sequences
> To tell our RNN to skip over padding tokens in our encoder
- need to tell Pytorch how long the actual(non-padded) sequences are
2. Masking
> Forces the model to ignore certain values, such as attention over
padded elements
3. Inference and BLEU

Bug For Pytorch 1.7: pack_padded_sequence requires lengths to be on CPU