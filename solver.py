import os
import time
import pickle
import copy
import numpy as np
from scipy.stats import norm
import random
from random import shuffle
import tqdm

import torch
import torch.nn as nn

import argparse
from tools import preprocessing

use_cuda=torch.cuda.is_available()

class SuperAwesomeNeuralNetwork(nn.Module):
      def __init__(self, input_dim, n_embeddings, n_layers=1):
          super(SuperAwesomeNeuralNetwork, self).__init__()
          self.input_dim=input_dim
          self.n_embeddings = n_embeddings
          self.embedding = nn.Embedding(n_embeddings+1,input_dim,max_norm=1)#one extra embedding for initial hidden state
          self.gru = nn.GRU(input_dim, input_dim, n_layers, bidirectional=False)
  
      def forward(self, events, input_lengths):
          #events: Var of shape (step(T), batch_size(B)), sorted decreasingly by lengths (because packing)
          #input_lengths: length of each sequence
          #returns: GRU outputs in form (T,B,hidden_size(H)), last output of GRU(1,B,H)
          if use_cuda:
              hidden = self.embedding(torch.LongTensor([self.n_embeddings]*events.size()[1]).cuda()).view(1,-1,self.input_dim)
          else:
              hidden = self.embedding(torch.LongTensor([self.n_embeddings]*events.size()[1])).view(1,-1,self.input_dim)
          embedded = self.embedding(events)
          packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
          output, hidden = self.gru(packed, hidden)
          output, output_length = torch.nn.utils.rnn.pad_packed_sequence(output)
          return output, hidden

def main():
    print("Get the data")
    n_data=100000
    checkpoint_name = "data_checkpoint_"+str(n_data)+".pickle"
    if os.path.isfile(checkpoint_name):
        all_data = pickle.load(open(checkpoint_name,'rb'))
    else:
        all_data = [a for a in tqdm.tqdm(preprocessing.career_trajectories(n_data, 'data/HiringPatterns.csv'),total=n_data)]
        pickle.dump(all_data,open(checkpoint_name,'wb'))
    name2id = all_data[0]
    id2name={}
    for key,value in name2id.items():
        id2name[value]=key
    data=[arr.astype(int).tolist() for arr in all_data[1:]]
    input_dim = 15
    n_embeddings = max(max(path) for path in data)+1
    batch_size = 32
    model = SuperAwesomeNeuralNetwork(input_dim,n_embeddings)
    if use_cuda:
        model.cuda()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    n_epochs = 5
    print("Training")
    for epoch in range(n_epochs):
        print("Epoch ",epoch+1)
        random.shuffle(data)
        for i in range(0,len(data),batch_size):
            batch = data[i:i+batch_size]
            batch.sort(key=lambda x:-len(x))
            input_lengths = np.array([len(b) for b in batch])
            batch = [torch.LongTensor(entry) for entry in batch]
            events = nn.utils.rnn.pad_sequence(batch, padding_value=0)
            optimizer.zero_grad()
            loss=0
            input_len, batch_size, *rest = events.size()
            if use_cuda:
                events=events.cuda()
            outputs, last_hidden = model(events,input_lengths)
            for j in range(batch_size):
                for k in range(1,input_lengths[j]):
                    loss-=torch.dot(model.embedding(events[k][j]).view(-1),outputs[k-1][j].view(-1))
                    negative_samples = [random.randint(0,n_embeddings) for _ in range(10)]
                    negative_samples = filter(lambda a:a!=events[k][j],negative_samples)
                    for neg_sample in negative_samples:
                        if use_cuda:
                            loss+=torch.dot(model.embedding(torch.LongTensor([neg_sample]).cuda()).view(-1),outputs[k-1][j].view(-1))
                        else:
                            loss+=torch.dot(model.embedding(torch.LongTensor([neg_sample])).view(-1),outputs[k][j].view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
            if (i//batch_size)%20==0:
                print("Step {}/{}\tLoss: {}".format(i,len(data),loss.data.item()))
        torch.save(model.state_dict(),"trained_model")
    print("Getting statistics for each institution")
    institutions={}
    for i in range(0,len(data),batch_size):
        batch = data[i:i+batch_size]
        batch.sort(key=lambda x:-len(x))
        input_lengths = np.array([len(b) for b in batch])
        batch = [torch.LongTensor(entry) for entry in batch]
        events = nn.utils.rnn.pad_sequence(batch,padding_value=0)
        input_len, batch_size, *rest = events.size()
        if use_cuda:
            events=events.cuda()
        outputs, last_hidden = model(events,input_lengths)
        for j in range(batch_size):
            for k in range(1,input_lengths[j]):
                matching=torch.dot(model.embedding(events[k][j]).view(-1),outputs[k-1][j].view(-1))
                if not events[k][j].data.item() in institutions:
                    institutions[events[k][j].data.item()]=[]
                institutions[events[k][j].data.item()].append(matching.data.item())
    inst_gauss={}
    for institution in institutions:
        inst_gauss[institution] = [np.mean(np.array(institutions[institution])),np.std(np.array(institutions[institution]))]
    print("Done training, evaluation")
    batch = data[:10]
    batch.sort(key=lambda x:-len(x))
    input_lengths = np.array([len(b) for b in batch])
    batch = [torch.LongTensor(entry) for entry in batch]
    events = nn.utils.rnn.pad_sequence(batch,padding_value=0)
    loss=0
    input_len, batch_size, *rest = events.size()
    if use_cuda:
        events=events.cuda()
    outputs, last_hidden = model(events,input_lengths)
    for j in range(10):
        print("\nPath {} starts in {}".format(j+1,id2name[events[0][j].cpu().data.item()]))
        for k in range(1,input_lengths[j]):
            matching=torch.dot(model.embedding(events[k][j]).view(-1),outputs[k-1][j].view(-1))
            print("Node {}, matching {}, node's distribution {}, probability: {}".format(id2name[events[k][j].cpu().data.item()],matching.data.item(),inst_gauss[events[k][j].data.item()],norm.cdf((matching.data.item()-inst_gauss[events[k][j].data.item()][0])/inst_gauss[events[k][j].data.item()][1])))

main()
