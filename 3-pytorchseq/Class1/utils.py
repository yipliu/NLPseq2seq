import torch
from tqdm import tqdm

def evaluate(model, tqdm_iterator, criterion, epoch, num_epochs):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in tqdm_iterator:

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()

            averaged_loss = epoch_loss / tqdm_iterator.total

            # update progress bar
            tqdm_iterator.set_description(f"Epoch [{epoch}/{num_epochs}]")
            tqdm_iterator.set_postfix(loss = averaged_loss)
        
    return averaged_loss

def train(model, tqdm_iterator, optimizer, criterion, clip, epoch, num_epochs):
    
    model.train()
    
    epoch_loss = 0

    for i, batch in tqdm_iterator:
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        # calculate the gradients
        loss.backward()
        
        # clip the gradients to prevent them from exploding 
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # update the parameters of our model
        optimizer.step()
        
        # sum the loss value to a running total
        epoch_loss += loss.item()

        averaged_loss = epoch_loss / tqdm_iterator.total
        # update progress bar
        tqdm_iterator.set_description(f"Epoch [{epoch}/{num_epochs}]")
        tqdm_iterator.set_postfix(loss = averaged_loss)
        
    return epoch_loss / tqdm_iterator.total
