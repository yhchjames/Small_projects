#%%
import pandas as pd

# %%
def isLs(s,t):
    
    rows = len(s)+1
    cols = len(t)+1
    dist = [[0 for x in range(cols)] for x in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i

    for i in range(1, cols):
        dist[0][i] = i
        
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 1
            dist[row][col] = min(dist[row-1][col] + 1,      # deletion
                                 dist[row][col-1] + 1,      # insertion
                                 dist[row-1][col-1] + cost) # substitution
 
    return dist[row][col] <=1

# %%
if __name__ == "__main__":
    df = pd.read_csv('20210103_hundenamen.csv')
    name_list = []
    for i in df['HUNDENAME']:
        if isLs('Luca',i) and i not in name_list:
            name_list.append(i)
    for name in name_list:
        print(name,end=';')