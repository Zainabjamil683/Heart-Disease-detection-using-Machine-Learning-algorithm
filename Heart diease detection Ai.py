#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"D:\heart-diease.csv")


# In[3]:


df


# In[4]:


df=df.rename(columns={"chol":"cholestrol"})
df=df.rename(columns={"fbs":"Fasting-Blood-Sugar"})
df=df.rename(columns={"Thalach":"max-heart-rate"})
df=df.rename(columns={"exang":"ex-induced-angina"})
df=df.rename(columns={"sex":"Gender"})
df=df.rename(columns={"cp":"chest-pain-type"})


# In[5]:


df.head()


# In[7]:


import sklearn.preprocessing

Label_Encoder =sklearn.preprocessing.LabelEncoder()
df['Gender']=Label_Encoder.fit_transform(df['Gender'])
df['Fasting-Blood-Sugar']=Label_Encoder.fit_transform(df['Fasting-Blood-Sugar'])
df['ex-induced-angina']=Label_Encoder.fit_transform(df['ex-induced-angina'])


# In[8]:


df.head()


# In[11]:


# Calculate correlation matrix
correlation_matrix = df.corr()


# In[10]:


# Plotting the heatmap
plt.figure(figsize=(7.5, 7.5))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# # Genetic as feature selection

# In[263]:


from sklearn.model_selection import train_test_split
import random


# In[264]:


x_input=df[['age','Gender','chest pain type','trestbps','cholestrol','Fasting-Blood-Sugar','restecg','thalach','ex-induced-angina','oldpeak','slope','ca','thal']]
y_output=df[['target']]


# In[265]:


X_train_heart, X_test_heart, Y_train_heart, Y_test_heart = train_test_split(x_input,y_output, test_size=0.3,random_state=42)


# In[266]:


def encode_chromosome(num_features):
    return [random.randint(0, 1) for _ in range(num_features)]

def select_parents(population, fitness_scores):
    return random.choices(population, weights=fitness_scores, k=2)

def crossover(parents, crossover_rate):
    child1, child2 = parents
    if random.random() < crossover_rate:
        crossover_point = random.randint(1, len(child1) - 1)
        child1[crossover_point:], child2[crossover_point:] = child2[crossover_point:], child1[crossover_point:]
    return child1, child2
def mutate(chromosome, mutation_rate):
    mutated_chromosome = chromosome.copy()
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = 1 - mutated_chromosome[i]
    return mutated_chromosome
POPULATION_SIZE = 10
def generate_initial_population(num_features, population_size):
    return [encode_chromosome(num_features) for _ in range(population_size)]
def evaluate_population(population, X_train_heart, Y_train_heart, num_features):
    fitness_scores = []
    for chromosome in population:
        fitness_scores.append(sum(chromosome))
    return fitness_scores
def generate_offsprings(population, fitness_scores, crossover_rate, mutation_rate):
    offsprings = []
    while len(offsprings) < len(population):
        parents = select_parents(population, fitness_scores)
        child1, child2 = crossover(parents, crossover_rate)
        
        # Apply mutation to the offspring
        child1 = mutate(child1, mutation_rate)
        child2 = mutate(child2, mutation_rate)
        
        offsprings.append(child1)
        offsprings.append(child2)
    return offsprings
def evaluate_population(population, X_train_heart, Y_train_heart, num_features):
    fitness_scores = []
    for chromosome in population:
        # Counting the number of ones in the chromosome (selected features)
        num_ones = sum(chromosome)
        fitness_scores.append(num_ones)
    return fitness_scores
def select_survivors(population, offsprings, fitness_scores):
    population_with_offsprings = population + offsprings
    sorted_indices = np.argsort(fitness_scores)[::-1]
    return [population_with_offsprings[i] for i in sorted_indices[:len(population)]]
def genetic_algorithm(X_train_heart, Y_train_heart, num_features, num_generations, crossover_rate,mutation_rate):
    #seed for reproducibility
    random.seed(42)  
    population = generate_initial_population(num_features, POPULATION_SIZE)
    for _ in range(num_generations):
        fitness_scores = evaluate_population(population, X_train_heart, Y_train_heart, num_features)
        offsprings = generate_offsprings(population, fitness_scores, crossover_rate,mutation_rate)
        population = select_survivors(population, offsprings, fitness_scores)
    best_chromosome = population[0]
    selected_features = [i for i, bit in enumerate(best_chromosome) if bit == 1]
    return selected_features


# In[267]:


num_generations =98
crossover_rate = 0.8
mutation_rate = 0.05
selected_features = genetic_algorithm(x_input, y_output, len(x_input.columns), num_generations, crossover_rate, mutation_rate)
print("Selected Features:", selected_features)


# In[268]:


df.head(1)


# # Decision tree as a classifier

# In[269]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[270]:


# Selecting only the important features obtained from genetic algorithm
x_input_selected = x_input.iloc[:, selected_features]
# Splitting the selected data into training and testing sets
X_train_selected, X_test_selected, Y_train, Y_test = train_test_split(x_input_selected, y_output, test_size=0.3, random_state=42)


# In[271]:


# Create Decision Tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)


# In[272]:


# Train the model
decision_tree.fit(X_train_selected, Y_train)


# In[273]:


# Predict on the test set
Y_pred_dt = decision_tree.predict(X_test_selected)


# In[274]:


# Accuracy on the test set
accuracy_dt = accuracy_score(Y_test, Y_pred_dt)
print(f"Accuracy of Decision Tree Classifier with selected features: {accuracy_dt * 100:.2f}%")


# In[275]:


# Compute confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred_dt)
print("Confusion Matrix:")
print(conf_matrix)


# # naive bayes

# In[276]:


from sklearn.naive_bayes import GaussianNB


# In[277]:


# Selecting only the important features obtained from genetic algorithm
x_input_selected = x_input.iloc[:, selected_features]
# Splitting the selected data into training and testing sets
X_train_selected, X_test_selected, Y_train, Y_test = train_test_split(x_input_selected, y_output, test_size=0.3, random_state=42)


# In[278]:


# Create Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()
# Train the model
nb_classifier.fit(X_train_selected, Y_train.values.ravel())
# Predict on the test set
Y_pred_nb = nb_classifier.predict(X_test_selected)


# In[279]:


# Accuracy on the test set
accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
print(f"Accuracy of Naive Bayes Classifier with selected features: {accuracy_nb * 100:.2f}%")


# In[280]:


# Creating a confusion matrix
conf_matrix_nb = confusion_matrix(Y_test, Y_pred_nb)
print("Confusion Matrix:")
print(conf_matrix_nb)


# # Knn as classifier

# In[281]:


from sklearn.neighbors import KNeighborsClassifier


# In[282]:


# Selecting only the important features obtained from genetic algorithm
x_input_selected = x_input.iloc[:, selected_features]
# Splitting the selected data into training and testing sets
X_train_selected, X_test_selected, Y_train, Y_test = train_test_split(x_input_selected, y_output, test_size=0.3, random_state=42)


# In[283]:


# Create KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=7)  # You can adjust the number of neighbors


# In[284]:


# Train the model
knn_classifier.fit(X_train_selected, Y_train.values.ravel())
# Predict on the test set
Y_pred_knn = knn_classifier.predict(X_test_selected)


# In[285]:


# Accuracy on the test set
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)
print(f"Accuracy of KNN Classifier with selected features: {accuracy_knn * 100:.2f}%")


# In[286]:


# Confusion matrix
conf_matrix_knn = confusion_matrix(Y_test, Y_pred_knn)
print("\nConfusion Matrix for KNN Classifier:")
print(conf_matrix_knn)


# # Comparision Between Classifiers

# In[287]:


import matplotlib.pyplot as plt
import numpy as np

# Accuracy values for each classifier
accuracy_values = [accuracy_dt, accuracy_nb, accuracy_knn]
classifiers = ['Decision Tree', 'Naive Bayes', 'KNN']

# Convert accuracy values to percentages
accuracy_values_percent = [value * 100 for value in accuracy_values]

# Plotting the horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(classifiers, accuracy_values_percent, color=['blue', 'green', 'orange'])
plt.xlim([0, 100])  # Set the x-axis limit between 0 and 100 for percentage values

# Adding labels and title
plt.xlabel('Accuracy (%)')
plt.ylabel('Classifiers')
plt.title('Classifier Comparison')

# Displaying the accuracy values on the right side of the bars
for i, value in enumerate(accuracy_values_percent):
    plt.text(value + 1, i, f'{value:.2f}%', ha='left', va='center')

# Show the plot
plt.show()


# # Age Anaylsis

# In[288]:


age_column = df['age']

# Bar Graph
plt.figure(figsize=(4, 4))
sns.histplot(age_column, bins=20, kde=False, color='skyblue', edgecolor='black')
plt.title('Distribution of Ages (Bar Graph)')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Pie Chart
plt.figure(figsize=(4, 4))
age_ranges = pd.cut(age_column, bins=[29, 39, 49, 59, 69, 79, 89], labels=['30-39', '40-49', '50-59', '60-69', '70-79', '80-89'])
age_range_counts = age_ranges.value_counts()
plt.pie(age_range_counts, labels=age_range_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Distribution of Ages (Pie Chart)')
plt.show()

Most people in the dataset are between 40 and 60 years old, showing that there are more middle-aged individuals 
compared to other age groups.
# # Chest Pain Type Analysis

# In[289]:


chest_pain_column = df['chest pain type']

# Count the occurrences of each chest pain type
chest_pain_counts = chest_pain_column.value_counts()

# Bar Graph
plt.figure(figsize=(5, 5))
sns.barplot(x=chest_pain_counts.index, y=chest_pain_counts.values, palette='viridis')
plt.title('Distribution of Chest Pain Types')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.show()

# Pie Chart
plt.figure(figsize=(5, 5))
plt.pie(chest_pain_counts, labels=chest_pain_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Distribution of Chest Pain Types')
plt.show()

People who are on 3rd level of chest pain are very less as compared to people who are on 2nd level of chest pain. 
We guess Most people died after 2nd level of chest pain
# ## Fasting Blood Sugar (Fbs) Analysis

# In[290]:


blood_sugar_column = df['Fasting-Blood-Sugar']

# Count the occurrences of each fasting blood sugar level
blood_sugar_counts = blood_sugar_column.value_counts()

# Bar Graph
plt.figure(figsize=(5, 5))
sns.barplot(x=blood_sugar_counts.index, y=blood_sugar_counts.values, palette='viridis')
plt.title('Distribution of Fasting Blood Sugar Levels')
plt.xlabel('Fasting Blood Sugar Level')
plt.ylabel('Count')
plt.show()

# Pie Chart
plt.figure(figsize=(5, 5))
plt.pie(blood_sugar_counts, labels=blood_sugar_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Distribution of Fasting Blood Sugar Levels')
plt.show()

People having fps < 120 have more chance of having Heart Disease than people havnig fps >120
# # Restecg Analysis

# In[291]:


restecg_column = df['restecg']

# Bar Graph
plt.figure(figsize=(5, 5))
sns.countplot(x=restecg_column, palette='viridis')
plt.title('Distribution of Resting Electrocardiographic Results (Bar Graph)')
plt.xlabel('Resting ECG Result')
plt.ylabel('Count')
plt.show()

# Pie Chart
plt.figure(figsize=(4, 4))
restecg_counts = restecg_column.value_counts()
plt.pie(restecg_counts, labels=restecg_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Distribution of Resting Electrocardiographic Results (Pie Chart)')
plt.show()

if resting electrocardiographic is 1 then person have more chances of suffering from Heart Disease
# # Exang (exercise - induced - angina) Analysis

# In[292]:


exang_column = df['ex-induced-angina']

# Bar Graph
plt.figure(figsize=(5, 5))
sns.countplot(x=exang_column, palette='viridis')
plt.title('Distribution of Exercise-Induced Angina (Bar Graph)')
plt.xlabel('Exercise-Induced Angina')
plt.ylabel('Count')
plt.show()

# Pie Chart
plt.figure(figsize=(5, 5))
exang_counts = exang_column.value_counts()
plt.pie(exang_counts, labels=exang_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Distribution of Exercise-Induced Angina (Pie Chart)')
plt.show()

If 'ex-induced-angina' is 0, there's a lower chance of chest pain during exercise and 
potentially less risk of heart disease.
# # The slope of the peak exercise ST segment (slope) Analysis

# In[293]:


slope_column = df['slope']

# Bar Graph
plt.figure(figsize=(4, 4))
sns.countplot(x=slope_column, palette='viridis')
plt.title('Distribution of Slope (Bar Graph)')
plt.xlabel('Slope')
plt.ylabel('Count')
plt.show()

# Pie Chart
plt.figure(figsize=(5, 5))
slope_counts = slope_column.value_counts()
plt.pie(slope_counts, labels=slope_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Distribution of Slope (Pie Chart)')
plt.show()

Feature (the peak exercise ST segment slope) has three symbolic values (flat, up sloping, downsloping)

Therefore People having up sloping are more prone to Heart Disease than flat and downsloping. 
# # number of major vessels colored by flourosopy (category)

# In[294]:


ca_column = df['ca']

# Bar Graph
plt.figure(figsize=(4, 4))
sns.countplot(x=ca_column, palette='viridis')
plt.title('Distribution of CA (Bar Graph)')
plt.xlabel('CA')
plt.ylabel('Count')
plt.show()

# Pie Chart
plt.figure(figsize=(4, 4))
ca_counts = ca_column.value_counts()
plt.pie(ca_counts, labels=ca_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Distribution of CA (Pie Chart)')
plt.show()

Since Fluoroscopy use to produce x-ray which will makes possible to see internal organs in motion.
Fluoroscopy uses x-ray to produce real-time video images
# # Thal Analysis

# In[295]:


thal_column = df['thal']

# Bar Graph
plt.figure(figsize=(5, 5))
sns.countplot(x=thal_column, palette='viridis')
plt.title('Distribution of Thal (Bar Graph)')
plt.xlabel('Thal')
plt.ylabel('Count')
plt.show()

# Pie Chart
plt.figure(figsize=(5, 5))
thal_counts = thal_column.value_counts()
plt.pie(thal_counts, labels=thal_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis'))
plt.title('Distribution of Thal (Pie Chart)')
plt.show()

type 2 thalassemia is the most common, followed by type 3. Type 1 is less frequent in the dataset.
# # Predicate logics
∃x(chest pain type(x)>0∧max heart rate(x)>150∧ST segment slope(x)>1→Heart Disease(x))

∀x(age(x)>50∧exercise-induced angina(x)=0∧thal(x)<2→Heart Disease(x))

∀x(resting ECG(x)=1∧exercise-induced angina(x)=1∧old peak(x)>2→Heart Disease(x))

∀x(exercise-induced angina(x)=0∧old peak(x)≤1→¬Heart Disease(x))

∃x(chest pain type(x)=0∧resting ECG(x)=0→¬Heart Disease(x))

∃x(thal(x)<2∧chest pain type(x)>0∧old peak(x)>1→Heart Disease(x))
# In[ ]:




