import csv
import numpy as np
import pandas as pd
import pyttsx3 
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing



# Chargement de données

data = pd.read_csv('DataBot/mydata.csv')
cols = data.columns
cols = cols[:-1]
X = data[cols]
Y = data['prognosis']

# Mapping des données
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(Y)
Y = label_encoder.transform(Y)
np.set_printoptions(threshold=np.inf)

# Division des données en deux parties 25% test et 75% training
speed = 42
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=speed, stratify=Y)

# Normalisation de données
"""
Les données sont normalisés et bien échantillées nous disposons de 4920 lignes et 133 colonnes au total
avec 75% (3690, 132) pour l'entrainement et 25% (1230, 132) pour le test du modèle.
"""

# Modélisation
dcl = DecisionTreeClassifier()
modele = dcl.fit(X_train, y_train)
score = cross_val_score(modele, X_test, y_test, cv=5, scoring= 'accuracy' )
print('\n','Scores : \t', score.mean())


# --- Programmation
description_list = {}
precaution_list = {}
severity_list = dict()

def getDescription():
    global description_list
    with open ('DataBot/symptom_description.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        try:
            _description_temp = {row[0]:row[1] for row in reader}
            description_list.update(_description_temp)
        except:
            pass

def getSeverity():
    global severity_list
    with open ('DataBot/symptom_severity.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter= ',')
        try:
            _severity_temp = {row[0]:int(row[1]) for row in reader}
            severity_list.update(_severity_temp)
        except:
            pass

def getPrecaution():
    global precaution_list
    with open ('DataBot/symptom_precaution.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        try:
            _precaution_temp = {row[0]:[row[1], row[2],row[3], row[4]] for row in reader}
            precaution_list.update(_precaution_temp)
        except :
            pass

def getDataInfoBot():
    getDescription()
    getPrecaution()
    getSeverity()

# Ensemble de symptômes presents dans le dataset
global_symptoms = [item for item in X]

# Prediction maladie
vect_zeros = []
def predict_by_symptom(symptoms_patient):
    global vect_zeros
    symptoms_data = {item : index for index, item in enumerate(X)}

    vect_zeros = np.zeros(len(symptoms_data)) # shape = 132,1 ---> 1, 132
    for item in symptoms_patient:
        if(item in global_symptoms):
            vect_zeros[symptoms_data[item]] = 1
    
    return modele.predict([vect_zeros])


#Lecture de text à haute voix 
def read_text(text):
    engine = pyttsx3.init()
    engine.setProperty('voice', 'english')  # choisir la voix anglaise
    engine.setProperty('rate', 150)  # régler la vitesse de la parole
    engine.say(text)  # dire la chaîne de caractères
    engine.runAndWait()
#read_text('should')

# Transformation inverse pour récupérer les maladie(s) predite(s)
def display(predict_values):
    prediction = label_encoder.inverse_transform(predict_values)
    for item in prediction:
        print("You have ", item)
        print(description_list[item],'\n')
        print('Take following precautions and consult a doctor quickly :\n')
        for value in precaution_list[item]:
            print('- ',value,'\n')

#Cette fonction prend une question en entrée et renvoie la réponse de l'utilisateur.
def get_user_answer(question):
    list_temp = []
    valid = True
    request = input('Choir un option ----->\t')
    if (request !='exit'):
        while valid:
            answer = input(question).strip().lower()
            if (len(list_temp)<4):
                list_temp.append(answer.split(","))
            else:
                print("Veuillez choisir une option valide parmi les choix proposés.")
                valid = False
    else:
        exit
    return list_temp

def predict_disease():
    # Interaction avec l'utilisateur
    question = "Quels sont les symptômes que vous ressentez ? (Séparez les symptômes par des virgules)\n"
    input_dict = get_user_answer(question)  
    if(len(input_dict) > 3):       
        # Prédiction de la maladie
        prediction = predict_by_symptom(input_dict)
        display(prediction)
    else:
        input_dict = get_user_answer(question)
    
    # Demande d'entrée utilisateur pour une nouvelle prédiction ou quitter

# Interface de binvenue
def get_info():
    print('\n<------------------Welcome to ManConsulTo-------------------->\n\n')
    print('Avant de vous consulter, faisons un petit point de sur notre conversation.\nEn effet, je suis ManConsulTo, chat bot basé sur l\'intelligence artificielle, \net je ne consulte les patients qu\'en anglais.\nMerci de saisir vos demandes en anglais !\n')
    name = input('Please, enter your name \t---->')
    print('Hello ', name,'\n')


getDataInfoBot()
#predict_disease()
list_test = ['palpitations','fatigue','loss_of_balance','high_fever']
one = ['muscle_weakness', 'high_fever', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'painful_walking']
#predict = predict_by_symptom(one)
#predict_symptom(one)
#display(predict)
check = ",".join(one).split(',')
print(check)
print('\n',type(check))

## c'est ici, qu'on doit appeler la fonction utilisant le modele de machine learning pour interagir avce l'utilisateur. 

print('<------------------Health care all right reserved 2023, Pensylvanie Valley-------------------->')

tsest = "e,x,i,t"
tsest = tsest.split(",")
print(tsest)
def tree_chat_bot():
    # Obtenir le chemin de décision pour chaque échantillon
    path = modele.decision_path([vect_zeros])
    print('\nPremiere path : \n',vect_zeros.shape[0], 'Bonjour','\n')
    # Convertir le chemin de décision en matrice creuse
    path = csr_matrix(path)
    print('Deuxieme path : \n',path,'\n')
    # Boucle sur chaque échantillon
    i = 0
    for i in range(len([vect_zeros])):
        node_indice = path[i].indices

        print(node_indice)
    print('\n Len vect zeros', len([vect_zeros]))
    for i in range(len([vect_zeros])):
        exp_symptoms = []
        # Extraire les nœuds traversés pour l'échantillon i
        node_indices = path[i].indices
        
        # Pour chaque nœud, poser une question à l'utilisateur
        for node_idx in node_indices:
            # Utiliser les caractéristiques associées au nœud pour poser une question
            feature_idx = modele.tree_.feature[node_idx]
            if feature_idx != -2:
                feature_name = cols[feature_idx]
                question = f"Est-ce que vous avez des symptômes de {feature_name.replace('_', ' ')}?"
                user_answer = get_user_answer(question)
                if (user_answer == 'yes'):
                    exp_symptoms.append(feature_name)
                elif (user_answer == 'no'):
                    break
                else:
                    print('Choose ---> no\yest')


  





