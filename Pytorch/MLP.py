import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics

# Implementiamo una rete neurale MLP (Multi-Layer Perceptron) per la classificazione delle immagini del dataset FashionMNIST.
# Il dataset FashionMNIST è un dataset di immagini di abbigliamento, composto da 60.000 immagini di addestramento e 10.000 immagini di test, 
# ciascuna delle quali è un'immagine in scala di grigi di 28x28 pixel.

# Importiamo il test e training set
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

# Utilizziamo l'hardware corretto per l'addestramento del modello
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
print(torch.version.hip)

#  La rete neurale MLP è composta da più strati di neuroni, ciascuno dei quali è connesso a tutti i neuroni dello strato successivo.
# La inizializziamo come classe che estende nn.Module, la classe base per tutti i modelli di rete neurale in PyTorch.
class OurMLP(nn.Module):
    def __init__(self):
        super().__init__()


        ''' Metodo con self.mlp

        # Definiamo la rete neurale MLP con 4 strati completamente connessi (fully connected layers).
        # Ogni strato è composto da un certo numero di neuroni, e utilizziamo la funzione di attivazione Sigmoid tra gli strati.
        # NOTA: Parti sempre da 1 hidden layer e aumenta il numero man mano in base all'accuratezza. Uguale per il numero di neuroni.

        self.mlp = nn.Sequential(
            # L'input layer ha 28*28 neuroni (corrispondenti a ciascun pixel dell'immagine) e 5 output
            nn.Linear(28*28, 15),
            #nn.Sigmoid(),
            nn.ReLU(),
            # Il secondo strato ha 5 neuroni in input e 5 neuroni in output
            # Rimuovomendo un layer aumenta l'accuratezza
            #nn.Linear(5, 5),
            #nn.Sigmoid(),
            #nn.ReLU(),
            # Il terzo strato ha 5 neuroni in input e 5 neuroni in output
            nn.Linear(15, 15),
            #nn.Sigmoid(),
            nn.ReLU(),
            # L'output layer ha 10 neuroni (corrispondenti alle 10 classi del dataset FashionMNIST).
            nn.Linear(15, 10)
        )
        self.flatten = nn.Flatten() '''
        
        ''' Metodo senza self.mlp'''
        # Definiamo la rete neurale MLP
        self.input_layer = nn.Linear(28*28, 15) # 28*28 neuroni in input e 15 neuroni in output
        self.hidden_layer1 = nn.Linear(15, 15) # 15 neuroni in input e 15 neuroni in output
        self.output_layer = nn.Linear(15, 10) # 15 neuroni in input e 10 neuroni in output (corrispondenti alle 10 classi del dataset FashionMNIST)
        self.activation = nn.ReLU() # Funzione di attivazione ReLU
        self.flatten = nn.Flatten() # Appiattiamo l'input (28x28) in un vettore di dimensione 784 (28*28)

    def forward(self, x):
        ''' Metodo con self.mlp
        x = self.flatten(x)
        logits = self.mlp(x)
        '''

        ''' Metodo senza self.mlp''' 

        x = self.flatten(x) 
        
        x = self.input_layer(x)
        x = self.activation(x)
        
        x = self.hidden_layer1(x)
        x = self.activation(x)

        logits = self.output_layer(x)
        
        return logits

# Modello
model = OurMLP()

# Numero diepoche = numero di iterazioni in cui il modello viene addestrato sui dati di addestramento.
epochs = 5

# La dimensione del batch è il numero di campioni che vengono elaborati insieme in un'unica iterazione.
batch_size = 64

# La learning rate è il tasso di apprendimento, che determina quanto i pesi del modello vengono aggiornati durante l'addestramento.
# Un valore troppo alto può portare a una convergenza instabile, mentre un valore troppo basso può rendere l'addestramento molto lento.
# Accuracy 0.1 quindi aumentiamo 
learning_rate = 0.001 # 0.0001

# La funzione di perdita (loss function) misura quanto il modello si discosta dalle etichette corrette durante l'addestramento.
loss_fn = nn.CrossEntropyLoss()

# Carichiamo i dati
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Funzione per il calcolo dell'accuratezza del modello
metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)

# Funzione di addestramento del modello si occupa di aggiornare i pesi del modello in base ai dati di addestramento.
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# Migliore optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def training_loop(dataloader, model, loss_fn, optimizer):
    dataset_size = len(dataloader)
    for batch, (X, y) in enumerate(dataloader):
        # Calcoliamo le previsioni del modello
        pred = model(X)

        # Calcoliamo la perdita
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 500 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss}, [{current}/{dataset_size}]")
            
            accuracy = metric(pred, y)
            print(f"Accuracy on current batch: {accuracy}\n")
    
    accuracy = metric.compute()
    print(f"--> Accuracy on training set: {accuracy}")
    metric.reset()

def testing_loop(dataloader, model, loss_fn):
    with torch.no_grad():
        for (X,y) in dataloader:
            pred = model(X)
            accuracy = metric(pred, y)

    accuracy = metric.compute()
    print(f"--> Accuracy on test set: {accuracy}")
    metric.reset()

for e in range(epochs):
    print("#-----------------------------#")
    print(f"\nEpoch {e}")
    training_loop(training_data_loader, model, loss_fn, optimizer)
    testing_loop(test_data_loader, model, loss_fn)

print("Training complete")
