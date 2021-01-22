# This class is to beautify the code using Thao stlye
>Data has been split using "processtext.py": 0.8 -> train; 0.2 -> text

## Novel function in this repo
- A tag for Cross Validation or not
- Plot the result
- [Set Default GPU in Pytorch](https://jdhao.github.io/2018/04/02/pytorch-gpu-usage/)

## The flow in this code
In Cross_Validation, The training data should be split in K fold: K-1 for train, 1 for eval(val).

# Encoder
- In the forward method, we pass in the source sentence, X, which is converted into dense vectors using the *embedding* layer

- As we pass a whole sequence to the RNN, it will automatically do the recurrent calculation of the hidden states over the whole sequence for us! Thus, hidden state (return of Encoder) is context vector Z ($h_{T}$)

- If no hidden state is passed to the RNN, it will automatically create an initial hidden state as a tensor of all zeros.