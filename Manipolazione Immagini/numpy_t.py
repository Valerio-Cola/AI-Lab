import array
l = list(range(1, 10))
A = array.array('i', l)

import numpy as np

# 1. Array 1D
x1 = np.array([1, 2, 3, 4, 5, 6], dtype='int32') # uint float  

# Riorganizza in una matrice 2x3
x1_grid = x1.reshape((2,3)) 
# Riorganizza in una matrice con conteggio automatico delle colonne
x1_grid = x1.reshape((3, -1))

# 2. Array 2D 
x2 = np.array([[1, 2, 3], 
               [4, 5, 6], 
               [7, 8, 9]])

x2.ndim     # Dimensione 
x2.shape    # Forma dell'array
x2.size     # Numero di elementi
x2.dtype    # Tipo di dati dell'array
x2.itemsize # Dimensione di ciascun elemento in byte

x2[0, :]    # Prima riga
x2[:, 0]    # Prima colonna
x2[:2, 1:3] # Prima due righe, seconda e terza colonna
x2[0,0]= 99 # Modifica elemento in posizione 0,0

# Vista Subarray con le prime due righe e colonne, modifica l'array originale 
x2_sub = x2[:2, :2] 
x2_sub[0, 0] = 42 
# In questo modo si crea una copia del subarray
x2_sub_copy = x2[:2, :2].copy() 

# Riorganizza in un array 1D
x2_linear = x2.reshape(1, -1) 

# 3. Array 3D
x3 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

# 4. Array preset
np.zeros((2, 3)) # 2x3 array di zeri
np.zeros((2, 3, 2), dtype='int32') # 2x3x2 array di zeri
np.ones((2, 3)) # 2x3 array di uno
np.full((2, 2), 99) # 2x2 array di 99

# 5. Array di numeri casuali
np.random.rand(4, 2) # 4x2
np.random.random((2, 2)) # 2x2
np.random.randint(1, 7, size=(3, 3)) # 3x3 array di interi casuali da 1 a 6
np.random.normal(0, 1, (3, 3)) # min = 0, stdev = 1, 3x3 array di numeri casuali dalla distribuzione normale
np.random.seed(0) # seed 

np.arange(1, 10, 2) # min, max, step [1, 3, 5, 7, 9]
np.linspace(0, 10, 5) # min, max, [0, 2.5, 5, 7.5, 10]
np.eye(3) # 3x3 matrice identità
np.empty(5) # 1D array di 5 valori non inizializzati, più veloce

# 6. Concatenazione di array
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
np.concatenate([x, y]) 

z = np.array([[7, 8, 9], 
              [10, 11, 12],
              [13, 14, 15]])

h = np.array([[16], 
              [17],
              [18]])

n = np.vstack([x, z]) # Aggiunge righe 
m = np.hstack([z, h]) # Aggiunge colonne

s = np.array([1,2,3,4,5,6,7,8,9])
# Crea 3 array splittando s negli indici 3 e 6
# [1, 2, 3] [4, 5, 6] [7, 8, 9]
s1, s2, s3 = np.split(s, [3, 6]) 

# Spezza la seconda riga
s1, s2 = np.vsplit(z, [2]) 
s1 = [[ 7,  8,  9],
      [10, 11, 12]]
s2 = [[13, 14, 15]]

# Spezza la seconda colonna
s1, s2 = np.hsplit(z, [2]) 
s1 = [[ 7, 8],
      [10, 11],
      [13, 14]]
s2 = [[ 9],
      [12],
      [15]]