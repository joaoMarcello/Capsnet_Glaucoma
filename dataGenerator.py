from keras.utils import Sequence
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt 

#------------------------------------------------------------------------------------------------
# Classe responsavel por gerar batches (um conjunto de imagens)
class DataGenerator(Sequence):
    # Construtor.
    # Recebe:
    #   * list_IDs: array com o caminho para cada uma das imagens.
    #   * labels: array com a classificacao das imagens.
    #   * batch_size: o tamanho do batch.
    #   * dim: dimensao das imagens.
    #   * n_channels: quantidade de canais.
    #   * n_classes: quantidade de classes.
    #   * shuffle: Se deve ou nao embaralhar os indices de list_IDs no final de cada epoca.
    def __init__(self, list_IDs, labels, batch_size = 32, dim = (224,224), n_channels = 3, n_classes = 2, shuffle = True, data = None, vessel=None):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.data = data
        self.vessel = vessel
        self.on_epoch_end()


    # Indica o numero de batches por epoca.
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    # Retorna um batch de imagens.
    # Recebe:
    #   * index: o indice do batch desejado.
    # Retorna:
    #   * X: array contendo n imagens, com n igual a batch_size.
    #   * y: classificacao da imagem
    def __getitem__(self, index):
        # Gerando array de indices
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Gerando o array com o caminho para as imagens a partir dos indices em indexes
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Gerando o batch
        X, y, v = self.__data_generation(list_IDs_temp)

        return [X, v, to_categorical(y, self.n_classes)], [to_categorical(y, self.n_classes), X]


    # Funcao responsavel por gerar os batches.
    # Recebe:
    #   list_IDs_temp: um array com o caminho para as imagens
    # Retorna:
    #   * X: array contendo n imagens, com n igual a batch_size.
    #   * y: classificacao da imagem.
    def __data_generation(self, list_IDs_temp):
        # Criando o array para as imagens na forma (batch_size, (224,224), 3 )
        X = np.zeros((self.batch_size, * self.dim, self.n_channels))
        # Criando o array para as classificacoes
        y = np.zeros((self.batch_size), dtype=int)

        v = np.zeros((self.batch_size, 112,112,1), 'float32')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Abrindo as imagens
            img = self.data[ID]

            # Adicionando a imagem no vetor de imagens
            X[i,] = img

            # Armazenado a classificacao da imagem
            y[i] = self.labels[ID]

            v[i] = self.vessel[ID]

        # Retornando o vetor de imagens(batch) e o vetor de classificacoes das imagens
        return X, y, v


    # Funcao chamada no final de cada epoca. Aqui apenas embaralhamos os indices de list_IDs 
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)