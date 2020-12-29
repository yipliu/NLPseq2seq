import torch
import torch.nn as nn

import time
import math
import spacy

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def model_train(model, iterator, optimizer, criterion, clip, epoch, num_epochs):

    # Set the model into "training mode"
    model.train()
    epoch_loss = 0

    for i, batch in iterator:
        """
        SRC =  Field(include_lengths = true), it makes batch.src = (tensor, the length)
        """
        src, src_len = batch.s
        trg = batch.t

        optimizer.zero_grad()

        output = model(src, src_len, trg) # [trg_len, batch_size, output_dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # update the parameters of our model
        optimizer.step()

        epoch_loss += loss.item()

        average_loss = epoch_loss / iterator.total

        # update progress bar
        iterator.set_description(f"Epoch [{epoch+1:02}/{num_epochs}]")
        iterator.set_postfix(loss = average_loss)

    return average_loss


def model_evaluate(model, iterator, criterion, epoch, num_epochs):
    
    # Set the model to evaluation mode
    model.eval()

    epoch_loss = 0

    # Ensure no gradients are calculated within the block
    with torch.no_grad():

        for i, batch in iterator:
            src, src_len = batch.s
            trg = batch.t

            output = model(src, src_len, trg, 0) # turn off teacher forcing

            # trg = [trg_len, batch_size]
            # output = [trg_len, batch_size, output_dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

            average_loss = epoch_loss / iterator.total

            # update
            iterator.set_description(f"Epoch [{epoch+1:02}/{num_epochs}]")
            iterator.set_postfix(loss = average_loss)
    
    return average_loss


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def translate_sentence(sentence, language_model, src_field, trg_field, model, device, max_len = 50):

    # 1. Ensure our model in evaluation mode, which it should always be for inference
    model.eval()
        
    """
    2. Ensure the sentence is tokenized. 
         if not, that means sentence is string, it should be tokenized

    
    """
    
    if isinstance(sentence, str):
        nlp = spacy.load(language_model)
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    # 3. numericalize the source sentence
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    """
    torch.LongTensor(src_indexes): make the src to a tensor, shape=[src_len]
    xx.unsqueeze(1): change the shape to [src_len, 1]
    """
    # 4. convert it to a tensor and add a batch dim
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)
    
    # 5. get src_len
    src_len = torch.LongTensor([len(src_indexes)]).to(device)
    
    with torch.no_grad():
        # 6. feed the source sentence into the encoder
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    # 7.create the mask for the source sentence
    mask = model.create_mask(src_tensor)
        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    # create a tensor to hold the attention values
    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        # store attention values
        attentions[i] = attention
        # get the predicted next token
        pred_token = output.argmax(1).item()
        # add prediction to current output sentence prediction
        trg_indexes.append(pred_token)
        # break if the prediction was an <eos> token
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    # convert the output sentence from indexes to tokens
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    # trg_tokens: <sos>, xxx, xxx,xx
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

def display_attention(sentence, translation, attention):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    attention = attention.squeeze(1).cpu().detach().numpy()
    
    cax = ax.matshow(attention, cmap='bone')
   
    ax.tick_params(labelsize=15)
    ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                       rotation=45)
    ax.set_yticklabels(['']+translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()