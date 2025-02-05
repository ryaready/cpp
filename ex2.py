import pandas as pd 
import numpy as np

terms1 = ['this', 'is', 'a', 'sample']
weights1 = [1, 1, 2, 1]

terms2 = ['this', 'is', 'another', 'example']
weights2 = [1, 1, 2, 3]



def compute_tfidf(terms1, weights1, terms2, weights2):
    df1 = pd.DataFrame({'Term': terms1, 'Term Count': weights1})
    df2 = pd.DataFrame({'Term': terms2, 'Term Count': weights2})
    
    total_terms_d1 = df1['Term Count'].sum()
    total_terms_d2 = df2['Term Count'].sum()
 
    df1['TF'] = df1['Term Count'] / total_terms_d1
    df2['TF'] = df2['Term Count'] / total_terms_d2
    
    df = pd.concat([df1, df2]).drop_duplicates(subset=['Term']).reset_index(drop=True)

    df['DF'] = df['Term'].apply(lambda term: ((term in terms1) + (term in terms2)))
    
    N = 2  
    df['IDF'] = np.log(N / df['DF'])
    
    df1 = df1.merge(df[['Term', 'IDF']], on='Term')
    df2 = df2.merge(df[['Term', 'IDF']], on='Term')
    
    df1['TF-IDF'] = df1['TF'] * df1['IDF']
    df2['TF-IDF'] = df2['TF'] * df2['IDF']
    
    return df1, df2


print(compute_tfidf(terms1, weights1, terms2, weights2))