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

#  La rete neurale MLP è composta da più strati di neuroni, ciascuno dei quali è connesso a tutti i neuroni dello strato successivo.
# La inizializziamo come classe che estende nn.Module, la classe base per tutti i modelli di rete neurale in PyTorch.
class OurMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Definiamo la rete neurale MLP con 4 strati completamente connessi (fully connected layers).
        # Ogni strato è composto da un certo numero di neuroni, e utilizziamo la funzione di attivazione Sigmoid tra gli strati.
        self.mlp = nn.Sequential(
            # L'input layer ha 28*28 neuroni (corrispondenti a ciascun pixel dell'immagine) e 5 output
            nn.Linear(28*28, 5),
            nn.Sigmoid(),
            # Il secondo strato ha 5 neuroni in input e 5 neuroni in output
            nn.Linear(5, 5),
            nn.Sigmoid(),
            # Il terzo strato ha 5 neuroni in input e 5 neuroni in output
            nn.Linear(5, 5),
            nn.Sigmoid(),
            # L'output layer ha 10 neuroni (corrispondenti alle 10 classi del dataset FashionMNIST).
            nn.Linear(5, 10)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits

# Modello
model = OurMLP()

# Numero diepoche = numero di iterazioni in cui il modello viene addestrato sui dati di addestramento.
epochs = 2

# La dimensione del batch è il numero di campioni che vengono elaborati insieme in un'unica iterazione.
batch_size = 64

# La learning rate è il tasso di apprendimento, che determina quanto i pesi del modello vengono aggiornati durante l'addestramento.
# Un valore troppo alto può portare a una convergenza instabile, mentre un valore troppo basso può rendere l'addestramento molto lento.
learning_rate = 0.0001

# La funzione di perdita (loss function) misura quanto il modello si discosta dalle etichette corrette durante l'addestramento.
loss_fn = nn.CrossEntropyLoss()

# Carichiamo i dati
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Funzione per il calcolo dell'accuratezza del modello
metric = torchmetrics.Accuracy(task="multiclass")

# Funzione di addestramento del modello si occupa di aggiornare i pesi del modello in base ai dati di addestramento.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

