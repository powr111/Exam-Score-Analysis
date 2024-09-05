import sqlite3 as sql, pandas as pd # handeling the data
import numpy as np, scipy.stats as stats # performing statistical analyis/mathmatics
import matplotlib.pyplot as plt # visulisations
from sklearn.model_selection import train_test_split # ML
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import os # used when debugging to fix an issue with the paths

df = pd.read_csv(r'C:\Users\alloh\OneDrive\Desktop\coding\workspace\CV_projs\SQL & ML\website\instance\exams.csv', encoding = 'latin-1')

df.rename(columns={'race/ethnicity':'ethnicity' ,
                   'parental level of education':'parents_education' ,
                   'test preparation course':'test_prep',
                   'math score':'math',
                   'writing score':'writing',
                   'reading score':'reading'},inplace=True)

with sql.connect(r'C:\Users\alloh\OneDrive\Desktop\coding\workspace\CV_projs\SQL & ML\website\instance\database.db') as conn: # cute bit of file handling
    df.to_sql('name', conn, if_exists = 'replace', index = False)
    # later we may have to use an f-string when the data becomes purely numerical
    try: # we use try here as it's a fun way to practice error handling
        # performing our query to the database
        result1 = pd.read_sql_query(
            f"SELECT AVG(\"math\") AS avg_math, MAX(\"math\") AS max_math, MIN(\"math\") AS min_math \
            FROM name \
            WHERE parents_education = 'associate''s degree' "
            , conn 
            )
        
        result2 = pd.read_sql_query(
            f"SELECT AVG(\"math\") AS avg_math, MAX(\"math\") AS max_math, MIN(\"math\") AS min_math \
            FROM name \
            WHERE parents_education = 'high school' OR parents_education = 'some high school' "
            , conn 
            )
        
        group_meals = {}
        for letter in range(65,70):
            group = f"group {chr(letter)}"
            result3 = pd.read_sql_query(
                f"SELECT COUNT(*) AS cheap_meal \
                FROM name \
                WHERE ethnicity = '{group}' AND lunch = 'free/reduced'"
                ,conn
            )
            group_meals[group] = result3.at[0,"cheap_meal"]
    except Exception as e:
        print(f"an error was found: {e}")

df_html = df[:10].to_html(classes='table table-striped', index=False)

aggragations_highschool = {'mean': round(result1.at[0,f"avg_math"],2),
        'maximum': result1.at[0,f"max_math"],
        'minimum': result1.at[0,f"min_math"],
         'Range': result1.at[0,f"max_math"] - result1.at[0,f"min_math"],
         'standard deviation': round(df[f"math"].std(),2)}

aggragations_associates = {'mean': round(result2.at[0,f"avg_math"],2),
        'maximum': result2.at[0,f"max_math"],
        'minimum': result2.at[0,f"min_math"],
         'Range': result2.at[0,f"max_math"] - result2.at[0,f"min_math"],
         'standard deviation': round(df[f"math"].std(),2)}

#figure our the problem with this cell
fig = plt.figure()
ax = fig.add_subplot()
ax.set_ylabel('Number of Students on meal plan')
ax.bar(list(group_meals.keys()), list(group_meals.values()), color=['green','red','blue','orange','yellow'])
plt.savefig(r'C:\Users\alloh\OneDrive\Desktop\coding\workspace\CV_projs\SQL & ML\website\statics\meals.jpeg')

fig = plt.figure()
ax = fig.add_subplot()
ax.hist(df['math'])
ax.set_xlabel('Values')
ax.set_ylabel('Frequency')
plt.savefig(r'C:\Users\alloh\OneDrive\Desktop\coding\workspace\CV_projs\SQL & ML\website\statics\maths.jpeg')

fig, (ax, ay) = plt.subplots(2, sharex=True)
# next we must convert our discrete data into a contineuous probability distribution.
frequincies = df['math'].value_counts(normalize=True)
ax.bar(frequincies.index,frequincies.values)

x = frequincies.index
y = frequincies.values

degree = 6 # we choose 6 as it provides a reasonably accurate approximation
coefficients = np.polyfit(x, y, degree)
polynomial = np.poly1d(coefficients)


x_curve = np.linspace(min(x), max(x), 500)
y_curve = polynomial(x_curve)

ay.bar(x, y, width=0.6, color='skyblue', edgecolor='black', label='Histogram')
ay.plot(x_curve, y_curve, color='red', label=f'Polynomial Degree {degree}')
ay.set_xlabel('Values')
ay.set_ylabel('Prbability Density')
ax.set_title('Histogram with and without curve')
fig.savefig(r'C:\Users\alloh\OneDrive\Desktop\coding\workspace\CV_projs\SQL & ML\website\statics\maths_hist.jpeg')

# performs a T-test on our data
test = stats.ttest_1samp(a = df['math'].values, popmean = aggragations_associates['mean'])

# checks if we can reject the null hypothesis
decide = stats.t.ppf(q = 0.025, df = 49)

# we may now begin the predictive analysis section of the project which will be used

# when doing this it's important to perform proper data cleaning and so we will be converting all data into numerical form and or dropping it
df['test_prep'] = df['test_prep'].replace('completed', 1)
df['test_prep'] = df['test_prep'].replace('none',0)

df['lunch'] = df['lunch'].replace('standard', 1)
df['lunch'] = df['lunch'].replace('free/reduced',0)


df['gender'] = df['gender'].replace('female',1)
df['gender'] = df['gender'].replace('male',0)

df = df.drop(columns = ['ethnicity','parents_education'])

X = df.drop(columns=['test_prep'])
y = df['test_prep']

# spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size=0.2)

scaler = MinMaxScaler(feature_range=(0,1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = knn.score(X_test,y_test)
