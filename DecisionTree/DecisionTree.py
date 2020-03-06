# Roll 17EC10003
# Name Anand Raj
# Assignment Number 1


import numpy as np
import pandas as pd
from numpy import log2 as log
import pprint
epis = np.finfo(float).eps

data_set = pd.read_csv('data1_19.csv')

#Train_data_set = data_set.sample(frac=0.75, random_state=99)  
#Test_data = data_set.loc[~data_set.index.isin(Train_data_set.index), :]
#Test_data = Test_data.reset_index(drop=True)
#Train_data_set = Train_data_set.reset_index(drop=True)
Data_Table=data_set
#table=Train_data_set #(As required)#


#This function returns the entropy of the current table
def Entropy_Table(table):
    Class=table.keys()[-1]
    entropy1=0
    values=table[Class].unique()
    for value in values:
        fraction=table[Class].value_counts()[value]/len(table[Class])
        entropy1+=-fraction*np.log2(fraction)
    return entropy1


#This function returns the entropy of the given attribute
def Entropy_Table_Attribute(table,attribute):
    Class=table.keys()[-1]
    target_variables=table[Class].unique()
    variables=table[attribute].unique()
    entropy2=0
    for variable in variables:
       entropy3=0
       for target_variable in target_variables:
           num = len(table[attribute][table[attribute]==variable][table[Class] ==target_variable])
           den = len(table[attribute][table[attribute]==variable])
           fraction=num/(den+epis)
           entropy3 += -fraction*log(fraction+epis)
       fraction2=den/len(table)
       entropy2 += -fraction2*entropy3
    return abs(entropy2)


#This function returns the best information gain attribte 
def Winner_Attribute(table):
    IG=[]
    for key in table.keys()[:-1]:
        IG.append(Entropy_Table(table)-Entropy_Table_Attribute(table,key))
    return table.keys()[:-1][np.argmax(IG)]

#This function returns the subtable
def Sub_Table(table,node,value):
    return (table[table[node]==value].reset_index(drop=True)).drop(node,axis=1)

#This function draws a tree as a dictionary
def Create_Tree(table,tree=None):
    at=table.columns
    if len(at)==1:
        return tree
    #Class=table.keys()[-1]
    node=Winner_Attribute(table)
    #print(node)
    att_values=np.unique(table[node])
    table1=table
    #print(table1)
    #print('Yo',node)
    if tree is None:
        tree={}
        tree[node]={}
    for value in att_values:
        #print(value)
        subtable=Sub_Table(table1,node,value)
        clValue,counts=np.unique(subtable['survived'],return_counts=True)
        #print(clValue)
        #print(counts)
        if len(counts)==1:
            tree[node][value]=clValue[0]
        elif counts[0]/counts[1]>100 or (len(at)==2 and counts[0]/counts[1]>1):#(According To Taste)
            tree[node][value]=clValue[0]
            
        elif counts[1]/counts[0]>100 or (len(at)==2 and counts[1]/counts[0]>1):#(According To Taste) 
            tree[node][value]=clValue[1]
        else:
            tree[node][value]=Create_Tree(subtable)
    
    return tree

Tree=Create_Tree(Data_Table)



#This function creates the prediction array
def Predict(data,tree):
    for key in tree.keys():
        
        value=data[key]
        
        if value not in tree[key]:
            return 'No'
        #Missing values
        
        tree=tree[key][value]
        #prediction = 0
        
        if type(tree) is dict:
            prediction = Predict(data,tree)
            
        else:
            prediction= tree
            break;
    
    return prediction



#Prediction array
pred = []

for i in range(len(data_set)):
    data=data_set.iloc[i]
    pred.append(Predict(data,Tree))
    


count=0
#Check score
for i in range(len(pred)):
    if pred[i]==data_set.survived[i]:
        count+=1
        
        
#A function to print the given tree 
def printTree(tree, d):
   
    for key in tree.keys():
        for key2 in tree[key].keys():
            print(key,' = ',key2)
            d+=1
            t2=tree[key][key2]
            
            for key3 in t2.keys():
                for key4 in t2[key3].keys():
                
                    for i in range(d):
                        print('|   ',end='')
                    print(key3,' = ',key4)
                    d+=1
                    t3=t2[key3][key4]
                    
                    for key5 in t3.keys():
                        for key6 in t3[key5].keys():
                            
                            for i in range(d):
                                print('|   ',end='')
                            print(key5,' = ',key6,'(',t3[key5][key6],')')
                    d-=1
            d-=1
    d-=1
        
#Print the tree            
printTree(Tree,0)

#Uncomment for accuracy        
print('\n\nPrediction Accuracy= ',(count*100)/len(pred),'%')        
     
    
            

            

