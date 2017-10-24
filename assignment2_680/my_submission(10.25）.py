'''

2017 IFN680 Assignment Two

Scaffholding code to get you started for the 2nd assignment.


'''

import random
import numpy as np

#import matplotlib.pyplot as plt

from tensorflow.contrib import keras

from tensorflow.contrib.keras import backend as K


import assign2_utils


#------------------------------------------------------------------------------

def euclidean_distance(vects):
    '''
    Auxiliary function to compute the Euclidian distance between two vectors
    in a Keras layer.
    '''
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

#------------------------------------------------------------------------------

def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    @param
      y_true : true label 1 for positive pair, 0 for negative pair
      y_pred : distance output of the Siamese network    
    '''
    margin = 1
    # if positive pair, y_true is 1, penalize for large distance returned by Siamese network
    # if negative pair, y_true is 0, penalize for distance smaller than the margin
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
#------------------------------------------------------------------------------

def compute_accuracy(predictions, labels):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    @param 
      predictions : values computed by the Siamese network
      labels : 1 for positive pair, 0 otherwise
    '''
    # the formula below, compute only the true positive rate]
    #    return labels[predictions.ravel() < 0.5].mean()
    n = labels.shape[0]
    acc =  (labels[predictions.ravel() < 0.5].sum() +  # count True Positive
               (1-labels[predictions.ravel() >= 0.5]).sum() ) / n  # True Negative
    return acc

#------------------------------------------------------------------------------

def create_pairs(x, digit_indices):
    '''
       Positive and negative pair creation.
       Alternates between positive and negative pairs.
       @param
         digit_indices : list of lists
            digit_indices[k] is the list of indices of occurences digit k in 
            the dataset
       @return
         P, L 
         where P is an array of pairs and L an array of labels
         L[i] ==1 if P[i] is a positive pair
         L[i] ==0 if P[i] is a negative pair
         
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            # z1 and z2 form a positive pair
            inc = random.randrange(1, 10)
            dn = (d + inc) % 10
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            # z1 and z2 form a negative pair
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

#------------------------------------------------------------------------------
    
def simplistic_solution():
    '''
    
    Train a Siamese network to predict whether two input images correspond to the 
    same digit.
    
    WARNING: 
        in your submission, you should use auxiliary functions to create the 
        Siamese network, to train it, and to compute its performance.
    
    
    '''
    def convolutional_neural_networks(input_dim):
        '''
        Base network to be shared (eq. to feature extraction).
        '''
        '''
        seq = keras.models.Sequential()
        seq.add(keras.layers.Dense(128, input_shape=(input_dim,), activation='relu'))
        seq.add(keras.layers.Dropout(0.1))
        seq.add(keras.layers.Dense(128, activation='relu'))
        seq.add(keras.layers.Dropout(0.1))
        seq.add(keras.layers.Dense(128, activation='relu'))
        '''
        seq = keras.models.Sequential()  #!!!arg keras.models.Sequential()
        seq.add(keras.layers.Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_dim))
        seq.add(keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'))
        seq.add(keras.layers.MaxPooling2D((2,2)))
        seq.add(keras.layers.Dropout(0.25))
        seq.add(keras.layers.Flatten())
        seq.add(keras.layers.Dense(128,activation='relu'))
        seq.add(keras.layers.Dropout(0.5))
        seq.add(keras.layers.Dense(10,activation='softmax'))
        return seq
        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
    # load the dataset
    #def original_data():
    x_train, y_train, x_test, y_test  = assign2_utils.load_dataset()

    input_dim = (28,28,1)

    x_train = x_train.reshape(x_train.shape[0], *input_dim).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], *input_dim).astype(np.float32)

    #x_train = x_train[:10000]
    #y_train = y_train[:10000]

    # Example of magic numbers (6000, 784)
    # This should be avoided. Here we could/should have retrieve the
    # dimensions of the arrays using the numpy ndarray method shape 

    x_train /= 255 # normalized the entries between 0 and 1
    x_test /= 255

    epochs = 2
    np.savez('original_data.npz', x_train = x_train, y_train = y_train)
    #return x_train,y_train,x_test,y_test,input_dim,epochs
    #x_train,y_train,x_test,y_test,input_dim,epochs = original_data()
    #'mnist_dataset.npz'
    def warped_data(x_train,y_train,file_name,rotation,variation):
        if(file_name == None):
            with np.load('original_data.npz') as npzfile:
                X_train = npzfile['x_train']
                y_train = npzfile['y_train']
            X_train = np.zeros_like(x_train)    
            for i in range(len(X_train[:,0])):
                X_train[i] = assign2_utils.random_deform(x_train[i],rotation,variation)
            np.savez(file_name, x_train = x_train, y_train = y_train)
        else:
            with np.load(file_name) as npzfile:
                X_train = npzfile['x_train']
                y_train = npzfile['y_train']
        return X_train,y_train
    # create training+test positive and negative pairs
    def train_test_Data(x_train,y_train,input_dim,epochs):
        digit_indices = [np.where(y_train == i)[0] for i in range(10)]
        #tr_pairs, tr_y = create_pairs(x_train, digit_indices)
        tr_pairs, tr_y = create_pairs(x_train, digit_indices)
        
        digit_indices = [np.where(y_test == i)[0] for i in range(10)]
        #te_pairs, te_y = create_pairs(x_test, digit_indices)
        te_pairs, te_y = create_pairs(x_test, digit_indices)
        # network definition
        base_network = convolutional_neural_networks(input_dim)
        
        input_a = keras.layers.Input(shape=input_dim)
        input_b = keras.layers.Input(shape=input_dim)
        
        # because we re-use the same instance `base_network`,
        # the weights of the network
        # will be shared across the two branches
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        
        # node to compute the distance between the two vectors
        # processed_a and processed_a
        distance = keras.layers.Lambda(euclidean_distance)([processed_a, processed_b])
        
        # Our model take as input a pair of images input_a and input_b
        # and output the Euclidian distance of the mapped inputs
        model = keras.models.Model([input_a, input_b], distance)
    
        # train
        rms = keras.optimizers.RMSprop()
        model.compile(loss=contrastive_loss, optimizer=rms)
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=128,
                  epochs=epochs,
                  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

        # compute final accuracy on training and test sets
        pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
        tr_acc = compute_accuracy(pred, tr_y)
        pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
        te_acc = compute_accuracy(pred, te_y)
    
        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    x_tr_W,y_tr_W = warped_data(x_train,y_train,'warped_all.npz',45,0.3)
    '''
    x_tr_W_E = np.zeros_like(x_train[:20000,])
    x_tr_W_H = np.zeros_like(x_train[20000:,])
    y_tr_W_E = np.zeros_like(y_train[:20000,])
    y_tr_W_H = np.zeros_like(y_train[20000:,])
    x_tr_W = np.zeros_like(x_train)
    y_tr_W = np.zeros_like(y_train)
    x_tr_W_E,y_tr_W_E = warped_data(x_tr_W_E,y_tr_W_E,'warped_E.npz',45,0.3)
    x_tr_W_H,y_tr_W_H = warped_data(x_tr_W_H,y_tr_W_H,'warped_H.npz',20,0.1)
    x_tr_W[:20000,], y_tr_W[:20000,] = x_tr_W_E,y_tr_W_E
    x_tr_W[20000:,],y_tr_W[20000:,] = x_tr_W_H,y_tr_W_H
    x_train = x_tr_W_E+x_tr_W_H
    y_train = y_tr_W_E+y_tr_W_H
    print(x_tr_W_E.shape)
    print(x_tr_W_H.shape)
    '''
    train_test_Data(x_tr_W,y_tr_W,input_dim,epochs)
    #train_test_Data(x_train,y_train,input_dim,epochs)

#------------------------------------------------------------------------------        


#------------------------------------------------------------------------------        
#------------------------------------------------------------------------------        

if __name__=='__main__':
    simplistic_solution()
    
    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
