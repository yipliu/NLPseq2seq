# a single GRU

import torch
import torch.nn as nn
import torch.nn.functional as F

import random

"""
For Encoder

"""

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout_p):
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = False)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, src, src_len):
        
        #src = [src_len, batch size] The value is token tensor. shape[0] = src_len[0]
        #src_len = [batch size] The value means the src length
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
                
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu())
                
        packed_outputs, hidden = self.rnn(packed_embedded)
                                 
        #packed_outputs is a packed sequence containing all hidden states, not tensor
        #hidden is now from the final non-padded element in the batch
        
        # Unpack the packed_outputs, return outputs and length of each
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs) 
            
        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros
            
        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        
        return outputs, hidden

"""
For attention

Previously: it pay attention to padding tokens within the source sentence

Now: Using masking force the attention to only be over non-padding elements
"""
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim  + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs, mask):
        
        #hidden = [1, batch size, dec hid dim]
        #encoder_outputs = [src_len, batch size, enc hid dim * 1]
        # mask.shape[batch_size, src_len]
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        hidden = hidden.repeat(src_len, 1, 1)
        # hidden = [src_len, batch_size, dec_hid_dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        """
        torch.cat() = [src_len, batchs_size, dec_hid_dim + enc_hid_dim]
        energy = [src_len, batch size, dec hid dim]
        """

        attention = self.v(energy).squeeze(2).permute(1,0)
        #attention = [batch_size, src_len]
        
        attention = attention.masked_fill(mask == 0, -1e10)
        """
        maked_fill: this will handle the tensor that its firt argument(mask == 0) is ture
                    It will give the value -1e10 to padding tokens
                    When the sentence is passed through the softmax, the padding tokens value will be zero
        """
        
        return F.softmax(attention, dim = 1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout_p, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(enc_hid_dim  + emb_dim, dec_hid_dim)
        
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout_p)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        """
        input = [batch_size]
        hidden = [1, batch_size, dec_hid_dim]
        encoder_outputs = [src_len, batch_size, enc_hid_dim]
        mask = [batch size, src len]
        """
        
        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch_size, emb_dim]
        
        attn_weights = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)            
        #attn_weights = [batch_size, 1, src_len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch_size, src_len, enc hid dim ]
        
        attn_applied = torch.bmm(attn_weights, encoder_outputs)
        #attn_applied = [batch size, 1, enc hid dim]
        
        attn_applied = attn_applied.permute(1, 0, 2)
        #attn_applied = [1, batch size, enc hid dim]
        
        rnn_input = self.dropout(torch.cat((embedded, attn_applied), dim = 2))
        #rnn_input = [1, batch size, enc_hid_dim  + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden)
        
        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        output = self.dropout(output)
        
        #output = F.log_softmax(self.fc_out(output[0]), dim=1)
        # [batch_size, dec_hid_dim]
        output = self.fc_out(output[0])
        # [batch_size, output_dim]
        return output, hidden, attn_weights.squeeze(1) # [batch_size, enc_hid_dim]






class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask
        
    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):
        
        """
        src = [src_len, batch_size]
        src_len = [batch_size]
        trg = [trg_len, batch_size]
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src, src_len)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        mask = self.create_mask(src)
        #mask = [batch size, src len]
                
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state, all encoder hidden states 
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            
        return outputs