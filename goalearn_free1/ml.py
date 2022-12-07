import joblib
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import math
import pickle


#part 1
print('part 1')
data=pd.read_excel('Goalearn-Data-Scientist-Mentoring-program-Mentor-Free-1-1.xlsx',sheet_name=1,header=[0,1])

data1=data[:8]['data link for 1-hiring'].drop('salary($).1',axis=1)

char_to_num={'five':5,'two':2,'seven':7,'three':3,'ten':10,'eleven':11}
data1['experience']=data1['experience'].dropna().replace(char_to_num)

data1.loc[data1['experience'].isnull() & (data1['salary($)']<50000),'experience']=1
data1.loc[data1['experience'].isnull() & (data1['salary($)']>=50000) & (data1['salary($)']<60000),'experience']=2
data1['test_score(out of 10)']=data1['test_score(out of 10)'].fillna(data1['test_score(out of 10)'].dropna().mean()//1)

x1=data1.drop('salary($)', axis=1)
y1=data1['salary($)']

lin_reg=LinearRegression()
lin_reg.fit(x1, y1)
test=[[2,9,6],[12,10,10]]
pred1=lin_reg.predict(test)
with open('part1.txt', 'w') as f:
    for i in range(len(pred1)):
        temp='predicted value for:'+str(test[i])+'\n'+str(pred1[i])+'\n'
        print(temp)
        f.write(temp)

print()

#part 2
print('part 2')

data2=data[:10]['data link 2-test_scores (1)'].drop(['name','cs.1'],axis=1)
x2=data2['math']
y2=data2['cs']

n = float(len(x2))
w=0
c=0
l=1e-4
prev_loss=float('inf')


while True:
    pred2 =w*x2+c
    loss=np.sum(np.power(pred2-y2,2))/n
    if(math.isclose(loss, prev_loss,rel_tol=1e-20)):
        print('w,bias,learning rate\n',w,'\t',c,'\t',l)
        pd.DataFrame(data=([[w,c,l]]),columns=['w','bias','learning rate'])\
            .to_csv('part2.csv',sep=',',encoding='utf-8')
        break
    D_w=(-2/n)*np.sum(x2*(y2-pred2))
    D_c=(-2/n)*np.sum(y2-pred2)
    w=w-l*D_w
    c=c-l*D_c
    prev_loss=loss

print()

#part 3
print('part 3')

data3=data[:13]['data link 3-test_scores (2)']


#pickle
print('pickle')
# x_feature=Mileage , y=Age(yrs)

x3_1_train, x3_1_test, y3_1_train, y3_1_test = model_selection.train_test_split(data3['Mileage'], data3['Age(yrs)'], test_size=0.33)
lin_reg3_1=LinearRegression()
lin_reg3_1.fit(x3_1_train.values.reshape(-1,1),y3_1_train)

filename='pickle_finalized_model3_1.sav'
pickle.dump(lin_reg3_1,open(filename,'wb'))


loaded_model=pickle.load(open(filename,'rb'))
result=loaded_model.score(x3_1_test.values.reshape(-1,1), y3_1_test)
print('result_3_1 (x=Mileage ,y=Age(yrs))\n',result)


# x_feature=Sell Price($) , y=Age(yrs)

x3_2_train, x3_2_test, y3_2_train, y3_2_test = model_selection.train_test_split(data3['Sell Price($)'], data3['Age(yrs)'], test_size=0.33)

lin_reg3_2=LinearRegression()
lin_reg3_2.fit(x3_2_train.values.reshape(-1,1),y3_2_train)

filename='pickle_finalized_model3_2.sav'
pickle.dump(lin_reg3_2,open(filename,'wb'))

loaded_model=pickle.load(open(filename,'rb'))
result=loaded_model.score(x3_2_test.values.reshape(-1,1), y3_2_test)
print('result_3_2 (x=Sell Price($) ,y=Age(yrs))\n',result)

print()
print()


#joblib
print('joblib')
# x_feature=Mileage , y=Age(yrs)
filename='joblib_finalized_model3_1.sav'
joblib.dump(lin_reg3_1,filename)

loaded_model=joblib.load(filename)
result=loaded_model.score(x3_1_test.values.reshape(-1,1), y3_1_test)
print('result_3_1 (x=Mileage ,y=Age(yrs))\n',result)

# x_feature=Sell Price($) , y=Age(yrs)
filename='joblib_finalized_model3_2.sav'
joblib.dump(lin_reg3_2,filename)

loaded_model=joblib.load(filename)
result=loaded_model.score(x3_2_test.values.reshape(-1,1), y3_2_test)
print('result_3_2 (x=Sell Price($) ,y=Age(yrs))\n',result)

print()
dummies=pd.get_dummies(data3['Car Model'],drop_first=True)
joined_data3=data3.join(dummies).drop('Car Model',axis=1)
print(joined_data3)
joined_data3.to_csv('part3.csv',sep=',',encoding='utf-8')
