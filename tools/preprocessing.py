import numpy as np
import pandas as pd


def clean(data, verbose=False):
    
    if verbose:
        print('--- Cleaning the dataset')

    # drop whitespace in columns
    data.columns = [''.join(col.split()) for col in data.columns.values]
    
    # rename columns
    data.rename(index=str,
                inplace=True,
                columns={"SkillRequirementsmet?": "SkillsMet",
                         "identityinformation": "Gender",
                         "Unnamed:7": "Ethnicity",
                         "Unnamed:8": "Disabled"})
    
    # drop some nasty columns
    data.drop(['ApplicantID', 'Top3softskillscandidateselectedfromfixedlist', 'Candidateinterviewed'], axis=1, inplace=True)
    
    # change the index to a integer
    data.index = data.index.astype(int)
    
    # drop the first line
    data.drop(0, axis=0, inplace=True)

    # remove duplicate values
    froms = ['NO', 'no', 'yes', np.nan, ' N/A', '   N/A', 'Female ']
    tos =   ['No', 'No', 'Yes', 'N/A',    'N/A',   'N/A', 'Female']
    data.replace(froms, tos, inplace=True)
    
    return data

def get_trajectory(data, verbose=False):
    
    if verbose:
        print('--- Getting trajectory')

    # sample a row
    idx = np.random.randint(1, len(data)+1)
    trj = data.loc[idx]
    
    # augment the row with some probability
    probability = 0.3
    for i in range(len(trj)):
        
        # determine the category
        category = trj.index[i]
        value = trj[i]
        
        if np.random.random() < probability:
            
            #print(category)
            choices = data[category].unique()
            new_value = np.random.choice(choices)
            #print('augmenting {} to {}'.format(value, new_value))
            trj[i] = new_value
    
    return trj

def add_school_info(trj, data, verbose=False):
    
    if verbose:
        print('--- Adding school information')

    # determine the school ranking (correlated to uni ranking)
    rankings = data['UniRanking'].unique()
    min_rank = np.min(rankings)
    max_rank = np.max(rankings)
    
    # get the jump
    rank_jump = np.random.poisson(5)
    rank_jump = np.random.choice([1, -1]) * rank_jump
    
    # get the school ranking
    uni_rank = trj.UniRanking
    school_rank = np.clip(uni_rank + rank_jump, min_rank, max_rank)
    
    # add it to the series
    trj.loc['SchoolRank'] = school_rank
    
    return trj


def sort_trajectory(trj, verbose=False):
    
    if verbose:
        print('--- Sorting trajectory')

    # order of events
    ordered = ['Gender',
               'Ethnicity',
               'Disabled',
               'SchoolRank',
               'UniRanking',
               'Departmentofstudy',
               'Internships',
               'SkillsMet',
               'Hired']
    
    # and their values
    vals = [trj[pt] for pt in ordered]
    
    return pd.Series(data=vals, index=ordered)

def get_encoder(trj, data, verbose=False):
    
    if verbose:
        print('--- Sorting encoder')

    encoder = {}
    state_code = 0
    
    for i in range(len(trj)):
        
        # get the category and the values
        category = trj.index[i]
        
        # skip school rank
        query_cat = category
        if category == 'SchoolRank':
            query_cat = 'UniRanking'

        # get different values
        try:
            values = sorted(data[query_cat].unique())
        except TypeError:
            values = data[category].unique()
        
        # encode
        for value in values:
            key = (category, value)
            encoder[key] = state_code
            state_code +=1
    
    return encoder


def encode_trajectory(trj, encoder, verbose=False):
    
    if verbose:
        print('--- Encoding trajectory')

    encoded_trj = np.zeros(len(trj))
    
    for i, col in enumerate(trj.index):
        state_code = encoder[(col, trj[col])]
        encoded_trj[i] = state_code
    
    return encoded_trj


def career_trajectories(N=None, datapath='data/HiringPatterns.csv', verbose=False):
    
    # get the data
    data = pd.read_csv(datapath)
    
    # clean it
    data = clean(data)
    
    for i in range(N):

        # generate an applicant
        trj = get_trajectory(data, verbose)
    
        # add school information
        trj = add_school_info(trj, data, verbose)
    
        # sort in the correct career trajectory
        trj = sort_trajectory(trj, verbose)

        # yield encoder first
        if i==0:
            encoder = get_encoder(trj, data, verbose)
            yield encoder
            
        # encode as a trajectory
        trj = encode_trajectory(trj, encoder, verbose)
        
        # keep generating otherwise
        yield trj


def main():

	gen = career_trajectories(10, '../data/HiringPatterns.csv', verbose=True)

	encoder = next(gen)
	for trj in gen:
	    print(trj)

if __name__ == '__main__':
	main()


