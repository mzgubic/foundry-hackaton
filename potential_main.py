import os,sys
import time
import pickle
import copy
import numpy as np
from scipy.stats import norm
import random
from random import shuffle
import tqdm
import argparse
import math

import torch
import torch.nn as nn

import argparse
from tools import preprocessing

use_cuda=torch.cuda.is_available()

from tkinter import *
import tkinter as ttk

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
    parser=argparse.ArgumentParser()
    parser.add_argument("--train", help="Do the training.", action='store_true')
    args = parser.parse_args()
    id2name=pickle.load(open("id2name.pickle",'rb'))
    n_embeddings = len(id2name)
    input_dim = 15
    model = SuperAwesomeNeuralNetwork(input_dim,n_embeddings)
    if args.train:
        n_data=100000
        batch_size = 32
        learning_rate = 1e-4
        n_epochs = 2
        checkpoint_name = "data_checkpoint_"+str(n_data)+".pickle"
        print("Getting the data")
        if os.path.isfile(checkpoint_name):
            all_data = pickle.load(open(checkpoint_name,'rb'))
        else:
            all_data = [a for a in tqdm.tqdm(preprocessing.career_trajectories(n_data, 'data/HiringPatterns.csv'),total=n_data)]
            pickle.dump(all_data,open(checkpoint_name,'wb'))
        data=[arr.astype(int).tolist() for arr in all_data[1:]]
        if use_cuda:
            model.cuda()
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
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
        model.load_state_dict(torch.load("trained_model"))
        print("Analysing statistics for each institution")
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
        pickle.dump(inst_gauss,open("institution_info.pickle",'wb'))
        print("Done training")
    model.load_state_dict(torch.load("trained_model"))
    if use_cuda:
        model.cuda()
    inst_gauss=pickle.load(open("institution_info.pickle",'rb'))
    print("The model is ready, starting the GUI")


    #GUI
    root = Tk()
    root.title("Potential")
    # Add a grid
    mainframe = Frame(root)
    mainframe.grid(column=0,row=0, sticky=(N,W,E,S),padx=20,pady=20)
    mainframe.columnconfigure(0, weight = 1)
    mainframe.rowconfigure(0, weight = 1)
    #mainframe.pack(pady = 100, padx = 100)

    Genders = {"N/A","Female","Male","Other","Prefer not to say"}
    GenderVar = StringVar(root)
    GenderMenu = OptionMenu(mainframe,GenderVar,*Genders)
    Label(mainframe,text="Gender of the applicant").grid(row=1,column=1)
    GenderMenu.grid(row=2,column=1)
    GenderVar.set('N/A')
    def change_gender(*args):
        pass
    GenderVar.trace('w',change_gender)

    Ethnicity = {"N/A","White","BME"}
    EthnicityVar = StringVar(root)
    EthnicityMenu = OptionMenu(mainframe,EthnicityVar,*Ethnicity)
    Label(mainframe,text="Ethnicity of the applicant").grid(row=3,column=1)
    EthnicityMenu.grid(row=4,column=1)
    EthnicityVar.set('N/A')
    def change_ethnicity(*args):
        pass
    EthnicityVar.trace('w',change_ethnicity)

    Disability = {"no","yes","Prefer not to say"}
    DisabilityVar = StringVar(root)
    DisabilityMenu = OptionMenu(mainframe,DisabilityVar,*Disability)
    Label(mainframe,text="Is the applicant disabled?").grid(row=5,column=1)
    DisabilityMenu.grid(row=6,column=1)
    DisabilityVar.set('no')
    def change_disability(*args):
        pass
    DisabilityVar.trace('w',change_disability)

    Schools = {value[1] for key,value in id2name.items() if value[0]=="School"}
    SchoolsVar = StringVar(root)
    SchoolsMenu = OptionMenu(mainframe,SchoolsVar,*Schools)
    Label(mainframe,text="Applicant's school").grid(row=7,column=1)
    SchoolsMenu.grid(row=8,column=1)
    SchoolsVar.set(id2name[17][1])
    def change_schools(*args):
        pass
    SchoolsVar.trace('w',change_schools)

    Universities = {value[1] for key,value in id2name.items() if value[0]=="University"}
    UniversitiesVar = StringVar(root)
    UniversitiesMenu = OptionMenu(mainframe,UniversitiesVar,*Universities)
    Label(mainframe,text="Applicant's University").grid(row=9,column=1)
    UniversitiesMenu.grid(row=10,column=1)
    UniversitiesVar.set(id2name[42][1])
    def change_universities(*args):
        pass
    UniversitiesVar.trace('w',change_universities)

    Departments = {value[1] for key,value in id2name.items() if value[0]=="Departmentofstudy"}
    DepartmentsVar = StringVar(root)
    DepartmentsMenu = OptionMenu(mainframe,DepartmentsVar,*Departments)
    Label(mainframe,text="Applicant's department of study").grid(row=11,column=1)
    DepartmentsMenu.grid(row=12,column=1)
    DepartmentsVar.set(id2name[67][1])
    def change_departments(*args):
        pass
    DepartmentsVar.trace('w',change_departments)


    def ComputeProbabilities():
        path=[]
        path.append({"N/A":3,"Female":2,"Male":1,"Other":5,"Prefer not to say":4}[GenderVar.get()])
        path.append({"N/A":8,"White":9,"BME":10}[EthnicityVar.get()])
        path.append({"no":12,"yes":13,"Prefer not to say":16}[DisabilityVar.get()])
        SchoolEntry = SchoolsVar.get()
        for key,value in id2name.items():
            if value[1]==SchoolEntry:
                path.append(key)
        UniEntry = UniversitiesVar.get()
        for key,value in id2name.items():
            if value[1]==UniEntry:
                path.append(key)
        DepartmentEntry = DepartmentsVar.get()
        for key,value in id2name.items():
            if value[1]==DepartmentEntry:
                path.append(key)
        path.extend([89,93,96])
        sys.stdout.flush()
        batch = [path]
        batch.sort(key=lambda x:-len(x))
        input_lengths = np.array([len(b) for b in batch])
        batch = [torch.LongTensor(entry) for entry in batch]
        events = nn.utils.rnn.pad_sequence(batch,padding_value=0)
        input_len, batch_size, *rest = events.size()
        if use_cuda:
            events=events.cuda()
        sys.stdout.flush()
        outputs, last_hidden = model(events,input_lengths)
        sys.stdout.flush()
        probabilities=[]
        names=[]
        for k in [3,4,8]:#range(1,input_lengths[0]):
            matching=torch.dot(model.embedding(events[k][0]).view(-1),outputs[k-1][0].view(-1))
            probabilities.append(norm.cdf((matching.data.item()-inst_gauss[events[k][0].data.item()][0])/inst_gauss[events[k][0].data.item()][1]))
            names.append(id2name[events[k][0].cpu().data.item()][1])
        probabilities=[i if not math.isnan(i) else 0.0 for i in probabilities]
        SchoolProbVar=StringVar()
        colours=['red','orange','yellow','green']
        SchoolProbLabel=Label(root, textvariable=SchoolProbVar,bg=colours[int(probabilities[0]*4)])
        SchoolProbVar.set("Probability of getting into\n {} is {:.2f}.".format(names[0],probabilities[0]*100))
        UniProbVar=StringVar()
        UniProbLabel=Label(root, textvariable=UniProbVar,bg=colours[int(probabilities[1]*4)])
        UniProbVar.set("Probability of getting into\n {} is {:.2f}.".format(names[1],probabilities[1]*100))
        HireProbVar=StringVar()
        HireProbLabel=Label(root, textvariable=HireProbVar,bg=colours[int(probabilities[2]*4)])
        HireProbVar.set("Probability of getting\n the job is {:.2f}.".format(probabilities[2]*100))
        SchoolProbLabel.grid(row=8,column=0)
        UniProbLabel.grid(row=10,column=0)
        HireProbLabel.grid(row=13,column=0)
        
    SubmitButton = Button(mainframe, text="Submit", command=ComputeProbabilities)
    SubmitButton.grid(row=13, column=1)
    root.mainloop()

main()
