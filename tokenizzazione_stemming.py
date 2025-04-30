import nltk

# Comodo per installare i pacchetti necessari per il Natural Language Toolkit (NLTK), da fare una sola volta inseriscilo in un file a parte
#nltk.download('omw-1.4')
#nltk.download('tagsets')
#nltk.download('punkt')
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger_eng')
#nltk.download('tagsets_json')

# Tokenization
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

example_text = "Hello world! This is a test sentence." 

# Divide il testo in frasi e parole
sent_tokenized = sent_tokenize(example_text)
print("Sentence Tokenized:", sent_tokenized)
word_tokenized = word_tokenize(example_text)
print("Word Tokenized:", word_tokenized)

# Crea un set di parole che non aggiungono significato alla frase
stop_words = set(stopwords.words('english'))
print("Stop words:", stop_words)

# Rimozione del "rumore" dalla frase, filtriamo le stopwords
filtered_list = []
for word in word_tokenized:
    if word.lower() not in stop_words:
        filtered_list.append(word)
print("Filtered List:", filtered_list)

# Stemming
from nltk.stem import PorterStemmer

# Processo di ridurre una parola alla sua radice (o "stem"), eliminando suffissi e variazioni morfologiche.
# Per esempio, "running", "ran" e "runs" verranno ridotte a "run".
# In questo caso utilizzo l’algoritmo di Porter.
stemmer = PorterStemmer()
stemmed_list = [stemmer.stem(word) for word in word_tokenized]
print("Stemmed List:", stemmed_list)

# POS Tagging
# Il Part-of-Speech Tagging è il processo di assegnare a ogni parola di un testo il suo ruolo grammaticale (sostantivo, verbo, aggettivo, ecc.).

tag = nltk.pos_tag(word_tokenized)
print("POS Tagging:", nltk.help.upenn_tagset())

# Lemmatization
# La lemmatizzazione è un processo simile allo stemming, ma più sofisticato.
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

lemm = lemmatizer.lemmatize('scarves')
x = lemmatizer.lemmatize('worst', pos='a')
print("Lemmatized:", lemm, x)
