from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as  sns
import matplotlib.pyplot as plt

# Scelgo le etichette che ci interessano, in questo caso indovinerà solo 
# topic religiosi, spazio e computer grafica
categories = ['talk.religion.misc', 'sci.space', 'comp.graphics']

# Carico il train test dal dataset fetch_20newsgroups 
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# Inizializzo modello  
model = make_pipeline(TfidfVectorizer(),MultinomialNB())

# Alleno
model.fit(train.data, train.target)

# Predizioni sul modello di test
labels = model.predict(test.data)

# Stampiamo la confusion matrix per verificare la correttezza
conf_matrix = confusion_matrix(test.target, labels)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',square=True, cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Input personalizzato 
s = "NASA launches new space telescope"
predicted = model.predict([s])
print(f"'{s}' is predicted to be in category: {train.target_names[predicted[0]]}")