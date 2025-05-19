'''
CNN (Convolutional Neural Network) è un tipo di rete neurale profonda utilizzata principalmente per l'analisi delle immagini. 

Le CNN sono progettate per riconoscere schemi e caratteristiche nelle immagini attraverso l'uso di filtri convoluzionali.

Input layer
    |
    |
    V
Convolutional layer: - utilizza filtri per estrarre caratteristiche dalle immagini analizzando i singoli pixel
                     - restituisce in output una matrice di caratteristiche (feature map)
                     - a differenza delle MLP, non è necessario appiattire l'input poichè ogni pixel non corrisponde a un neurone 
    |
    |
    V                 
Pooling layer: - riduce la dimensione della matrice di caratteristiche (feature map) mantenendo le informazioni più importanti
                 in questo modo si riduce la dimensione dell'input, non abbiamo più un immagine ma un insieme di feature (matrice di valori)
    |
    |
    V
Fully connected layer:  - è l'equivalente di un multi layer perceptron
                        - ogni feature è connessa a tutti i neuroni del primo strato
                        - ogni neurone è connesso a tutti i neuroni dello strato successivo
                        - è utilizzato per l'ottimizzazione dei pesi e la classificazione finale

Hyperparameters: 
    - learning rate: velocità di apprendimento
    - batch size: numero di campioni da elaborare in un singolo passaggio
    - epochs: numero di passaggi attraverso l'intero dataset
    - optimizer: algoritmo per l'ottimizzazione dei pesi
    - filter dimension: dimensione del filtro convoluzionale, bisogna specificarlo per ogni convolutional layer
                        F x F x C: dimensione del filtro (F x F) e numero di canali (C)
                        I x I x C: dimensione dell'immagine (I x I) e numero di canali (C)
                        O x O x K (k filtri): dimensione dell'output (feature map O x O) e numero di filtri (K)
    - stride: numero di pixel da saltare durante la convoluzione
    - padding: numero di pixel da aggiungere ai bordi dell'immagine per mantenere le dimensioni
               in questo modo è possibile analizzare le immagini interamente senza perdere informazioni
               Valid padding: non viene aggiunto alcun padding
               Same padding: aggiunto per analizzare l'immagine senza perdere informazioni
               Full padding: aggiunto per analizzare ogni pixel dell'immagine 
    - convolution output size: O = [(I - F + P_start + P_end) / S] + 1            
'''

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
import torchmetrics

# Importiamo il test e training set
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Definiamo la rete neurale CNN
class OurCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Definiamo i layer della rete neurale CNN
        # Per ottimizzare meglio aumentare il numero di filtri out_channels->input_channels
        self.cnn = nn.Sequential(
            # Convolutional layer: 1 canale in input (immagine in scala di grigi), 10 filtri, dimensione del filtro 3x3
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3),
            nn.ReLU()
        )

        self.mlp = nn.Sequential(
            # Valore stampato in r90, 10 neuroni in output
            nn.Linear(5760, 15), 
            nn.ReLU(),
            
            # 10 neuroni in input e 10 neuroni in output (corrispondenti alle 10 classi del dataset FashionMNIST)
            nn.Linear(15, 10)
        )

        self.flatten = nn.Flatten() # Appiattiamo l'input

    def forward(self, x):
        # Passiamo l'input attraverso i layer CNN
        x = self.cnn(x) 
        
        # Appiattiamo l'input 
        x = self.flatten(x) 
        
        # Stampa la dimensione dell'input da dare alla MLP
        # print(x.shape)
        
        logits = self.mlp(x)
        return logits
    
model = OurCNN()        
# Spostiamo il modello sulla GPU se disponibile
# model = OurCNN().to(device) 

# Creiamo un input fittizio per verificare le dimensioni dell'input
# 1 immagine in input (batch size 1), 1 canale (immagine in scala di grigi), 28x28 dimensione dell'immagine
# fake_input = torch.rand(1, 1, 28, 28)
# model(fake_input)

# torch.Size([1, 5760])
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x5760 and 11520x10)
# Quindi la dimensione dell'input da dare alla MLP è 5760 e non 11520

# Definiamo gli iperparametri
epochs = 2
batch_size = 64
learning_rate = 0.001 

loss_fn = nn.CrossEntropyLoss()

# Carichiamo i dati
training_data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# Funzione per il calcolo dell'accuratezza del modello
metric = torchmetrics.Accuracy(task="multiclass", num_classes=10)

# Spostiamo la metrica sulla GPU se disponibile
# metric = metric.to(device) 

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
            # X = X.to(device)   Spostiamo i dati sulla GPU se disponibile
            # y = y.to(device)   Spostiamo i dati sulla GPU se disponibile
            pred = model(X)
            accuracy = metric(pred, y)

    accuracy = metric.compute()
    print(f"--> Accuracy on test set: {accuracy}")
    metric.reset()

for e in range(epochs):
    print("#-----------------------------#")
    print(f"\n[Epoch {e}]")
    training_loop(training_data_loader, model, loss_fn, optimizer)
    testing_loop(test_data_loader, model, loss_fn)

print("Training complete")

# Salvo solo i pesi
# torch.save(model.state_dict(), "modello_test_weights.pth")

# Salvo l'intero oggetto modello
torch.save(model, "modello_test.pth")