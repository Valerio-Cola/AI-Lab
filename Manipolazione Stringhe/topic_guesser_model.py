from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import seaborn as  sns
import matplotlib.pyplot as plt

# Data
categories = ['talk.religion.misc', 'sci.space', 'comp.graphics']

# Dataset
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

model = make_pipeline(TfidfVectorizer(),MultinomialNB())

model.fit(train.data, train.target)

labels = model.predict(test.data)

conf_matrix = confusion_matrix(test.target, labels)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',square=True, cbar=False, xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

s = "NASA launches new space telescope"
predicted = model.predict([s])
print(f"'{s}' is predicted to be in category: {train.target_names[predicted[0]]}")