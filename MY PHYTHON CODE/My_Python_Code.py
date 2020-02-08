
# coding: utf-8

# In[1]:


import pyodbc as od
import pandas as pd
conn = od.connect('Driver={SQL Server};Server=DESKTOP-C949P6F;Database=Voyager;Trusted_Connection=yes;')
rs = conn.cursor()
arrsize = 10

start = 'Select'
table = 'dw.AccountDimTest'
sel_list = ' * '
from_cl =  'from ' 
sql = start+sel_list+from_cl+table

rs.execute(sql).fetchmany(arrsize)

cols = [column[0] for column in rs.description]
mylist=[]

while True:
    rows = rs.fetchmany(arrsize)
    if not rows:
        break
    df = pd.DataFrame([tuple(t) for t in rows], columns = cols)
    mylist.append(df)

df = pd.concat(mylist, axis=0).reset_index(drop=True)

print(df['Account'])

# for row in rs:
#     print(row)

for i in range(2):
    print('hello')


# In[5]:


from turtle import *
shape('turtle')
for i in range(4):
    forward(100)
    right(90)


# In[4]:


'first degree equation'
def equation(a,b,c,d):
    return (d-b)/(a-c)

equation(.5,.66,.25,.87)


# In[12]:


from math import sqrt
'quadratic equation'
def quad(a,b,c):
    x1=(-b + sqrt(b**2 - 4*a*c))/(2*a)
    x2=(-b - sqrt(b**2 - 4*a*c))/(2*a)
    return(x1,x2)
quad(2,7,-15)


# In[ ]:



   


# In[3]:


def apply_discount(product, discount):
    price = int(product['price']) * (1.0 - discount)
    assert 0 <= price <= product['price']
    return price

shoes = {'name':'Fancy Shoes','price':14900}

apply_discount(shoes,0.25)
    
    


# In[10]:


import numpy as np
import pandas as pd
import re

df = pd.read_table('C:/temp/test.txt', delim_whitespace=True, names=('year', 'pop','name'),
                   dtype={'year': np.int, 'pop': np.int,'name':np.str})

#print(df)
#print(df[['year','name']])
#print(df[['year','name','pop']])
#print(df[['name']])
#print(df.loc[:13])   
#------------------------------
#print(df.groupby('year')['pop'].mean())
#fx = df.groupby('year')['pop'].mean()
#fx.plot()
#----------------------------------
# the concept of a dataframe is the  = of a dict
# which is the equivalent of the following:
# date conversion: born_dt = pd.to_datetime(mypeeps['born'],format='%Y-%m-%d')

years=[1957,1934]
target_years=[1957,1943]
names=['Fred','Mike']

def mtch(myyears,targ):
    matchlist=[]
    matches=0
    
    for i in range(len(targ)):
        if myyears[i]==targ[i]:
            matches+=1
            matchlist.append(myyears[i])
            
    return matchlist



mtch(years, target_years)


mypeeps = pd.DataFrame(
        data={'year':[1957,1934,1967],
               'pop':[3456,6765,9878],
                'name':['Fred','Mike','Sam']})

#mypeeps_f=mypeeps[mypeeps.name.isin(names) & mypeeps.year.isin(years)]
mypeeps_f=mypeeps[~mypeeps.year.isin(years)]
#mypeeps_f1=mypeeps_f[(mypeeps_f['pop'] > 5000)]
#--mypeeps_f[(mypeeps_f['pop'] > 5000 | mypeeps_f['year'] == 1957)]


print(mypeeps_f[['year','name','pop']])

#-------------joins in Python---------------------------------------
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]})
df3 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
#------------------------------------------------------------------


df40 = pd.merge(df1, df2, on='employee', how='inner')
df50 = pd.merge(df40,df3, on='group', how='inner')
df50[['group','supervisor']]
#------------------------------------ Regular Expression Matching ---
df51=df50[df50.supervisor.str.contains(r'St')]
#http://www.regexlab.com/wild2regex

print(df51)

#----------  filter a groupby the equivalent of having in sql 
def filter1(x):
    return x['hire_date'].sum() > 2012


# Define the aggregation calculations
aggregations = {
#     'duration': { # work on the "duration" column
#         'total_duration': 'sum',  # get the sum, and call this result 'total_duration'
#         'average_duration': 'mean', # get mean, call result 'average_duration'
#         'num_calls': 'count'
#     },
#     'date': {     # Now work on the "date" column
#         'max_date': 'max',   # Find the max, call the result "max_date"
#         'min_date': 'min',
#         'num_days': lambda x: max(x) - min(x)  # Calculate the date range per group
#     },
    'hire_date': ["count", "max","sum"]  # Calculate two results for the 'network' column with a list
}

# Perform groupby aggregation by "month", but only on the rows that are of type "call"
# data[data['item'] == 'call'].groupby('month').agg(aggregations)


# df60=df50.groupby('group').sum()
# df70 =df50.groupby('group').filter(filter1) 
# df80=df50.groupby(['group', 'supervisor'])['hire_date'].sum()
df80=df50.groupby(['group', 'supervisor']).agg(aggregations)
#df90=df80[df80.group.str.contains(r'St')]
#df70['EmployeeGroup'] = df70.employee + df70.group


# print(df50)
# print(df60)
# print(df70)
print(df80)
#print(mypeeps_f1)

#-----------------------------------------------------------------

#decade = 10 * (planets['year'] // 10)
#decade = decade.astype(str) + 's'
#decade.name = 'decade'
#planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)

#df90 = df80.groupby(['group','supervisor'], as_index=False).sum()

#Print(df90)



                


# In[7]:


#--- List comparisons ----
a = [1, 2, 3, 4, 5]
b = [9, 8, 7, 6, 5]

r2=set(a) & set(b)
r3=set(a) - set(b)
r4=set(a) | set(b)
r5= [i for i in a if i in b]


print(r2)
print(r3)
print(r4)
print(r5)


# In[1]:



def handle_error() :
    print('Error occured ')
    sys.exit(1)


#arrival: list containing the arrival time of the trains
#departure: list containing the departure time of the trains
#Return value: minimum number of train platforms needed 
def find_min_platforms(arrival, departure) :
    n = len(arrival)
    if (n == 0):
        return 0

    #Sort the arrival and departure time independently in non-decreasing order
    arrival.sort()
    departure.sort()

    cur_num_platforms = min_num_platforms = 1
    i = 1 #i is used for traversing arrival
    j = 0 #j is used for traversing departure

    while (i < n and j < n) :
        if (arrival[i] < departure[j]) :
            #A new train is coming in before a train departs. So  
            #we need an extra platform
            cur_num_platforms += 1
            if (cur_num_platforms > min_num_platforms):
                min_num_platforms = cur_num_platforms
            i += 1
        elif (arrival[i] == departure[j]):
            #A train arrives at the same time as a departing train. 
            #So we don't need an extra platform
            i += 1
            j += 1
        else :
            #A train departs before the new train arrives. 
            #So a platform is freed up
            cur_num_platforms -= 1
            j += 1
        
    return min_num_platforms





def test01() :
    arrival = [800, 900, 945, 1300, 1500, 1530, 1545] 
    departure = [1030, 915, 1100, 1400, 1545, 1830, 1715]

    print('Arrival: ', end='')
    print(arrival)

    print('Departure: ', end='')
    print(departure)

    result = find_min_platforms(arrival, departure)

    print('Minimum number of platforms = {}'.format(result) )

    expected_result = 2

    if (result != expected_result):
        handle_error()

    print('__________________________________')





if (__name__ == '__main__'):
    test01()
    print('Test passed')




# In[2]:



import random

MAX_NUM_TESTS = 10
MAX_NUM_ELEMENTS = 10
MAX_VALUE = 100


def handle_error() :
    print('Error occured')
    sys.exit(1)


#stock_price: list of stock price values
#Return value: maximum profit possible
def find_max_profit(stock_price) :
    n = len(stock_price)

    max_profit = 0
    if (n <= 1):
        return max_profit

    min_stock_price = stock_price[0]

    for  i in range(1, n):
        cur_profit = stock_price[i] - min_stock_price

        if (cur_profit > max_profit):
            max_profit = cur_profit

        if (stock_price[i] < min_stock_price):
            min_stock_price = stock_price[i]
    
    return max_profit






def find_brute_force_max_profit(stock_price) :
    n = len(stock_price)

    max_profit = 0
    if (n <= 1):
        return max_profit

    for  i in range(n - 1):
        for  j in range(i+1, n ):
            if (stock_price[j] > stock_price[i]) :
                cur_profit = stock_price[j] - stock_price[i]
                if (cur_profit > max_profit):
                    max_profit = cur_profit


    return max_profit




if (__name__ == '__main__'):
    for  test_nr in range(MAX_NUM_TESTS):
        #Randomly pick the number of elements
        num_elements = random.randint(1, MAX_NUM_ELEMENTS)

        #Add random share values to the list
        a = [random.randint(0, MAX_VALUE) for  i in range(num_elements)]

        print('Input : ', end='')
        print(a)

        #Find the best profit possible
        result = find_max_profit(a)

        print('Maximum profit = {}'.format(result) )

        #Find the best profit using the brute force approach
        brute_force_result = find_brute_force_max_profit(a)

        #Both results should match
        if (result != brute_force_result):
            handle_error()

        print('__________________________________________________')




    print('Test passed')









# In[3]:



class Activity(object):

    def __init__(self,  input_id, start_time, end_time):
        self.id = input_id
        self.start_time = start_time
        self.end_time = end_time
    


def handle_error() :
    print('Error occured ')
    sys.exit(1)




#a: list of activities, where each activity has a start time and end time
#Return value: list having the index of the selected activities 
def activity_selection(a) :
    #Sort the activities in non-decreasing order of their end time
    a.sort(key = lambda x: x.end_time)

    selected = [] 

    #Keep a track of the current time as we process the activities
    cur_time = 0

    for  i, cur_activity in enumerate(a):
        #Pick the activity whose start time is on or after current time
        if (cur_activity.start_time >= cur_time) :
            selected.append(i)

            #Update the current time to the end time of the activity
            cur_time = cur_activity.end_time

    return selected





def test01() :
    a = []

    obj = Activity(1000, 0, 5)
    a.append(obj)

    obj = Activity(1001, 1, 2)
    a.append(obj)

    obj = Activity(1002, 3, 6)
    a.append(obj)

    selected = activity_selection(a)

    for  index in selected:
        print('Perform Activity : {}, Start time = {}, End time = {} '.format(a[index].id,
                a[index].start_time, a[index].end_time) )
    

    expected_result = 2

    if (len(selected) != expected_result):
        handle_error()

    print('__________________________________')




def test02() :
    a = []

    obj = Activity(1000, 0, 1)
    a.append(obj)

    obj = Activity(1002, 1, 5)
    a.append(obj)

    obj = Activity(1001, 2, 3)
    a.append(obj)

    obj = Activity(1003, 4, 7)
    a.append(obj)

    selected = activity_selection(a)

    for index in selected:
        print('Perform Activity : {}, Start time = {}, End time = {} '.format(a[index].id,
                a[index].start_time, a[index].end_time) )


    expected_result = 3

    if (len(selected) != expected_result):
        handle_error()

    print('__________________________________')




if (__name__ == '__main__'):
    test01()
    test02()

    print('Test passed')




# In[4]:



import sys
import random


MAX_NUM_TESTS = 100
MAX_NUM_STATIONS = 10
MAX_DISTANCE = 100

def handle_error() :
    print('Error occured ')
    sys.exit(1)



#gas: the amount of gas available at each gas station. The total gas in all 
#   stations should be sufficient to complete the circular trip 
#distance: distance[i] has the distance between gas station i and i+1
#mileage: how much distance can the car travel for 1 unit of gas consumed
#Return value: station from where to start so that we don't run out of fuel and
#   complete the circular trip around all stations
def find_starting_gas_station(gas, distance, mileage) :
    num_stations = len(gas)
    assert(num_stations)

    #Station from where to start the journey so that we don't run out of fuel
    starting_station = 0 

    least_gas = 0 #Tracks the least amount of gas in fuel tank
    gas_in_tank = 0 #Tracks how much fuel is currently present in fuel tank
    for  i, (gas_in_station, cur_distance) in enumerate(zip(gas, distance)):
        gas_required = cur_distance // mileage
    
        #At station i, we fill up gas_in_station and then as we drive,  
        #we consume gas_required to reach the destination station = 
        #(i+1) % num_stations 
        gas_in_tank += gas_in_station - gas_required 
        if (gas_in_tank < least_gas) :
            least_gas = gas_in_tank
            #The starting station is the station where we have
            #the least amount of gas in the tank just before we fill up
            starting_station = (i+1) % num_stations
        
    return starting_station


#Verifies if we start at starting_station, we can complete the journey without running out of fuel
def verify(starting_station, gas, distance, mileage) :
    num_stations = len(gas)

    #Check if starting_station is out of range
    if (starting_station < 0 or starting_station >= num_stations):
        handle_error()

    cur_station = starting_station
    gas_in_tank = 0
    for  i in range(1, num_stations):
        gas_required = distance[cur_station] // mileage
        gas_in_tank += gas[cur_station] - gas_required

        #gas in the fuel tank should always be >= 0
        if (gas_in_tank < 0) :
            handle_error()
        
        cur_station = (cur_station + 1) % num_stations
    




def test() :

    #Randomly pick the number of gas stations
    num_stations = random.randint(1, MAX_NUM_STATIONS)

    gas = [0] * num_stations
    distance = [random.randint(1, 10) for  i in range(num_stations)] 

    total_distance = sum(distance)

    #We are fixing the mileage to 1 mile/gallon since we will
    #not have to deal with fractional values
    mileage = 1

    #Compute the gas needed to complete the journey around all stations
    total_gas = total_distance // mileage

    #Randomly distribute the total_gas among the gas stations
    remaining_gas = total_gas
    per_station_quota = remaining_gas // num_stations
    for  i in range(num_stations):
        if (remaining_gas > 0):
            gas[i] = random.randint(0, per_station_quota - 1)
        else :
            gas[i] = 0 
        remaining_gas -= gas[i]
     

    #If there is any gas left over, then distribute the 
    #remaining gas equally among the gas stations
    i = 0
    per_station_quota = remaining_gas // num_stations
    while (remaining_gas > 0 and i < num_stations - 1) :
        gas[i] += per_station_quota
        remaining_gas -= per_station_quota
        i += 1
    
    #If there is still any gas left over, give it to the last gas station
    gas[num_stations - 1] += remaining_gas 

    print('Gas      : ', end='')
    print(gas)


    print('Distance : ', end='')
    print(distance)

    #Find the gas station from where to start the journey
    #IMPORTANT: ensure that while calling this function that the sum of gas in all
    #the stations should be sufficient to complete the journey
    starting_station = find_starting_gas_station(gas, distance, mileage)

    print('Starting station = {}'.format(starting_station) )

    verify(starting_station, gas, distance, mileage)

    print('____________________________________________________')





if (__name__ == '__main__'):
    for  i in range(MAX_NUM_TESTS):
        test()

    print('Test passed')




# In[11]:


years=[1957,1934]
target_years=[1957,1943]
names=['Fred','Mike']

def mtch(myyears,targ):
    matchlist=[]
    matches=0
    
    for i in range(len(targ)):
        if myyears[i]==targ[i]:
            matches+=1
            matchlist.append(myyears[i])
            
    return matchlist



mtch(years, target_years)



# In[2]:


#histogram plot
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel("C:/temp/First.xlsx","Sheet1")

fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.hist(df['Age'],bins=7)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Employee')
plt.show


# In[7]:


#Scatterplot
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel("C:/temp/First.xlsx","Sheet1")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['Age'],df['Sales'])
plt.show()


# In[9]:





# In[2]:


#BoxPlot
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_excel("C:/temp/First.xlsx","Sheet1")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.boxplot(df['Age'])
plt.show()


# In[3]:


#ViolinPlot
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df=pd.read_excel("C:/temp/First.xlsx","Sheet1")

sns.violinplot(df['Age'],df['Gender'])
sns.despine


# In[4]:


#BubblePlot
#import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import seaborn as sns
df=pd.read_excel("C:/temp/First.xlsx","Sheet1")

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['Age'],df['Sales'],s=df['Income'])
plt.show()


# In[5]:


#LineChart
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
import pandas as pd
import seaborn as sns


var=df.groupby('BMI').Sales.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('BMI')
ax1.set_ylabel('Sum of Sales')
ax1.set_title('BMI by Sum of Sales')
var.plot(kind='line')


# In[19]:


from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X=heights,y=weights)

heights = [[1.6],[1.65],[1.7],[1.73],[1.8]]
weights = [[60],[65],[72.3],[75],[80]]

plt.title('Weights plotted against heights')
plt.xlabel('Heights in meters')
plt.ylabel('Weights in kilos')

plt.plot(heights,weights,'k.')

plt.axis([1.5,1.85,50,90])

plt.plot(heights,model.predict(heights))


# In[7]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import svm
import seaborn as sns; sns.set(font_scale=1.2)



df=pd.read_excel("C:/temp/First.xlsx","Sheet2")
df_pr=pd.read_excel("C:/temp/First.xlsx","Sheet3")



sns.lmplot('size','price',data=df,hue='sold',palette='Set2',fit_reg=False,scatter_kws={"s":50});

x = df[['size', 'price']].values
x_pr = df_pr[['size', 'price']].values
y = np.where(df['sold']=='y',1,0)
model = svm.SVC(kernel='linear').fit(x,y)

plt.xlabel('Size of House')
plt.ylabel('Asking price in thousands')
plt.title('Size of house and their asking price')

def will_it_sell(size,price):
    if(model.predict([[size,price]]))==0:
        print('Will not sell')
    else:
        print('Will sell')
        
        
will_it_sell(2500,400)
will_it_sell(2500,200)
        
        
                
                

    




# In[14]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set(font_scale=1.2)

#def fruit_predict():

data=pd.read_excel("C:/temp/First.xlsx","Sheet4")
   
training_set, test_set = train_test_split(data, test_size = 0.2, random_state = 1)

X_train = data[['Weight', 'Size']].values
Y_train = data[['Class']].values
X_test = data[['Weight', 'Size']].values
Y_test =  data[['Class']].values

classifier = SVC(kernel='rbf', random_state = 1)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

for yo, yp in zip(data['Class'].values,Y_pred):
  print(yo,yp)


#fruit_predict

