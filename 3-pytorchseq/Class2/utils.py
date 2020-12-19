import torch
import torch.nn as nn

import time
import math

def model_train(model, iterator, optimizer, criterion, clip, epoch, num_epochs):

    # Set the model into "training mode"
    model.train()
    epoch_loss = 0

    for i, batch in iterator:
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

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
            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) # turn off teacher forcing

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