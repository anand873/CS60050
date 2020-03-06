# Roll 17EC10003
# Name Anand Raj
# Assignment Number 2

import pandas as pd

#Data Input and Preprocessing
data = pd.read_csv('data2_19.csv')
test = pd.read_csv('test2_19.csv')
data = data["D,X1,X2,X3,X4,X5,X6"].str.split(",", n = 6, expand = True) 
test = test["D,X1,X2,X3,X4,X5,X6"].str.split(",", n = 6, expand = True)
   
data=data.astype(int)
test=test.astype(int)

data.columns=['D','X1','X2','X3','X4','X5','X6']
test.columns=['D','X1','X2','X3','X4','X5','X6']

#Dividing the test data
test_X = test.iloc[:,1:]
test_y = test.iloc[:,0]

labels = [1,2,3,4,5]

#Function to generate probability
def count(data,colname,label,target):
    condition = (data[colname] == label) & (data['D'] == target)
    return len(data[condition])

#Final Prediction Array
Prediction = []

#Probability Tree
Tree = {0:{},1:{}}

#Total 0 and 1
N0 = data['D'][data['D'] == 0].count()
N1 = data['D'][data['D'] == 1].count()

#Total entries
total = data['D'].count()

#Initial Probabilities
P_0 = N0/total
P_1 = N1/total

#Populating the Probability Tree
for col in data.columns[1:]:        
    Tree[0][col] = {}
    Tree[1][col] = {}
    for category in labels:
        count_ct_0 = count(data,col,category,0)
        count_ct_1 = count(data,col,category,1)
        Tree[0][col][category] = (count_ct_0+1)/(N0+5) #Using Laplacian Smoothing
        Tree[1][col][category] = (count_ct_1+1)/(N1+5) #Using Laplacian Smoothing


#Creating the Prediction Array
for row in range(0,len(test_X)):
    prod_0 = P_0
    prod_1 = P_1
    for feature in test_X.columns:
        prod_0 *= Tree[0][feature][test_X[feature].iloc[row]]
        prod_1 *= Tree[1][feature][test_X[feature].iloc[row]]
        
    #Predict the outcome
    if prod_0 > prod_1:
        Prediction.append(0)
    else:
        Prediction.append(1)

#Calculating the Accuracy       
tp,tn = 0,0
for j in range(0,len(Prediction)):
    if Prediction[j] == 0:
        if test_y.iloc[j] == 0:
            tp += 1
    else:
        if test_y.iloc[j] == 1:
            tn += 1
            
print('Accuracy On Test Data=',((tp+tn)/len(test_y))*100,"%")