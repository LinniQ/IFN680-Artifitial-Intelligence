'''
2017 IFN680 Assignment Two - Siamese Network

Group 9:
    Linni Qin n9632981
    
    
'''
#------------------------------------------------------------------------------

import random
import numpy as np
import matplotlib.pyplot as plt
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
       
def initial():
    '''
    Load the origianl data will calling assing2_utils.
    Save them into npz file1 (original_data.npz).
    
    '''
    x_train, y_train, x_test, y_test  = assign2_utils.load_dataset()      


    i_row,i_column = x_train.shape[1:3]
    channel = 1
    input_dim = (i_row,i_column,channel)
    
    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    # convert to float 32
    x_train = x_train.reshape(x_train.shape[0], *input_dim).astype(np.float32)
    x_test = x_test.reshape(x_test.shape[0], *input_dim).astype(np.float32)

    # normalized the entries between 0 and 1
    x_train /= 255 
    x_test /= 255
    
    #save the file into original_data.npz
    np.savez('original_data.npz', x_train = x_train, y_train = y_train,x_test = x_test,y_test = y_test)
    file1 ='original_data.npz'
    return input_dim,file1
    

#------------------------------------------------------------------------------
    
def warped_data(rotation,variation,input_dim,warped_size,file_name):

    '''
    Create file2 (warped_data.npz) to save the 100,000 warped sample images if required 
    with using method random_deform from assing2_util
    
    The parameter to warp the image will be inputted with testing purposes accordingly in exp2() and exp3()    
         
    '''
    
    # loading two return values in the method of initial()
    input_dim,file1 = initial()
    with np.load(file1) as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']
    
    # create 100,000 samples of warped images by using function random_deform in file assign2_util
    # random choosing from the original 60,000 images
    # to enhance the probobility of choosing each images, the other 40,000 is chosen reversedly from the dataset
    X_train = np.zeros((warped_size,*input_dim))
    for i in range(60000):
        X_train[i] = assign2_utils.random_deform(x_train[i],rotation,variation)
    for j in range(40000):
        X_train[60000-1 + j] = assign2_utils.random_deform(x_train[random.randint(0,60000-1)],rotation,variation)
        
    #save the file into warped_data.npz        
    np.savez('warped_data.npz', x_train = X_train, y_train = y_train, x_test = x_test, y_test = y_test)
    file2 = 'warped_data.npz'
    return file2

#------------------------------------------------------------------------------
  
def convolutional_neural_networks(y_train,input_dim):
    '''
    Create CNN:
        3 conventional layers and 2 fully connected layers
        
    '''
    num_class = len(np.unique(y_train))
    seq = keras.models.Sequential() 
    seq.add(keras.layers.Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_dim))
    #seq.add(keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'))
    seq.add(keras.layers.MaxPooling2D((2,2)))
    seq.add(keras.layers.Dropout(0.25))
    seq.add(keras.layers.Flatten())
    #seq.add(keras.layers.Dense(128,activation='relu'))
    #seq.add(keras.layers.Dropout(0.5))
    seq.add(keras.layers.Dense(num_class,activation='relu'))
    return seq

#------------------------------------------------------------------------------

def exp1():
    '''
    Experiment 1:
    Train the siamese network on the original dataset
    
    '''
    # file 1 is original_data.npz
    input_dim,file1 = initial()
    with np.load(file1) as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']

    #create paired images with using funciton create_pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(len(np.unique(y_train)))]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)    
    digit_indices = [np.where(y_test == i)[0] for i in range(len(np.unique(y_test)))]
    te_pairs, te_y = create_pairs(x_test, digit_indices)
    
    # network definition
    base_network = convolutional_neural_networks(y_train, input_dim)
    
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

    # train the model
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        batch_size=128,
                        epochs=epochs,
                        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))
   
    # evaluate the model
    pred1 = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred1, tr_y)    
    pred2 = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred2, te_y)

    # print the training and text accuracy 
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    #plot the change of the loss for each eposch   
    plt.plot(history.history['loss'],'r--')
    plt.plot(history.history['val_loss'], 'g--')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()     
     
#------------------------------------------------------------------------------

def exp2():
    '''
    Experiment 2:
    Train the siamese network on the warped dataset with hard degree (45 degree, 0.3 strength)
    
    '''
    input_dim,file1 = initial()
    
    # call method warped_data to warp the 100,000 image 
    # with Hard level (rotating 45 degree with 0.3 strength)
    file2 = warped_data(45, 0.3, input_dim, 100000,file1)
    with np.load(file2) as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']

    #create paired images with using funciton create_pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(len(np.unique(y_train)))]    
    tr_pairs_H, tr_y_H = create_pairs(x_train, digit_indices)    
    digit_indices = [np.where(y_test == i)[0] for i in range(len(np.unique(y_test)))]    
    te_pairs_H, te_y_H = create_pairs(x_test, digit_indices)
    
    # network definition
    base_network = convolutional_neural_networks(y_train, input_dim)
    
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

    # train the model
    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    history = model.fit([tr_pairs_H[:, 0], tr_pairs_H[:, 1]], tr_y_H,
                        batch_size=128,
                        epochs=epochs,
                        validation_data=([te_pairs_H[:, 0], te_pairs_H[:, 1]], te_y_H))
    
    # evaluate the model
    pred1 = model.predict([tr_pairs_H[:, 0], tr_pairs_H[:, 1]])
    tr_acc = compute_accuracy(pred1, tr_y_H)    
    pred2 = model.predict([te_pairs_H[:, 0], te_pairs_H[:, 1]])
    te_acc = compute_accuracy(pred2, te_y_H)

    # print the training and test accuracy
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    #plot the change of the loss for each eposch
    plt.plot(history.history['loss'],'r--')
    plt.plot(history.history['val_loss'], 'g--')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()     
    
#------------------------------------------------------------------------------

def exp3():
    '''
    Experiment 3:
    Train the siamese network on the significant easier dataset of warped images with 15 degree rotation
    and a projective transormation with a "strength" of 0.1.
    Then train the network on the hard dataset (45 degree, 0.3 strength)
    
    '''   
    
    input_dim,file1 = initial()
    

    # call method warped_data to warp the 100,000 image 
    # with Easy level (rotating 15 degree with 0.1 strength)
    # the rotation degree and parameter for strength can be changed if needed within experimental 3
    file_E = warped_data(15, 0.1, input_dim, 100000,file1)
    with np.load(file_E) as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']  

    #create Easy paird images with using method create_pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(len(np.unique(y_train)))]    
    tr_pairs_E, tr_y_E = create_pairs(x_train, digit_indices)    
    digit_indices = [np.where(y_test == i)[0] for i in range(len(np.unique(y_test)))]    
    te_pairs_E, te_y_E = create_pairs(x_test, digit_indices)
    
    # call method warped_data to warp the 100,000 image 
    # with Hard level (rotating 45 degree with 0.3 strength)
    # the rotation degree and parameter for strength can be changed if needed within experimental 3
    file_H = warped_data(45, 0.3, input_dim, 100000,file1)
    with np.load(file_H) as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']
        x_test = npzfile['x_test']
        y_test = npzfile['y_test']

    #create Hard paird images with using funciton create_pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(len(np.unique(y_train)))]
    tr_pairs_H, tr_y_H = create_pairs(x_train, digit_indices)    
    digit_indices = [np.where(y_test == i)[0] for i in range(len(np.unique(y_test)))]
    te_pairs_H, te_y_H = create_pairs(x_test, digit_indices)
    
    # network definition
    base_network = convolutional_neural_networks(y_train, input_dim)
    
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

    rms = keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    
    #train the model on Easy dataset
    history_E = model.fit([tr_pairs_E[:, 0], tr_pairs_E[:, 1]], tr_y_E,
                        batch_size=128,
                        epochs=epochs,
                        validation_data=([te_pairs_E[:, 0], te_pairs_E[:, 1]], te_y_E))  
    
    #train the model on Hard dataset
    history_H = model.fit([tr_pairs_H[:, 0], tr_pairs_H[:, 1]], tr_y_H,
                        batch_size=128,
                        epochs=epochs,
                        validation_data=([te_pairs_H[:, 0], te_pairs_H[:, 1]], te_y_H))
    
    #evaluate the model on Hard dataset
    pred1 = model.predict([tr_pairs_H[:, 0], tr_pairs_H[:, 1]])
    tr_acc = compute_accuracy(pred1, tr_y_H)
    
    pred2 = model.predict([te_pairs_H[:, 0], te_pairs_H[:, 1]])
    te_acc = compute_accuracy(pred2, te_y_H)

    #print the training and test accuracy
    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    #plot the change of the loss for each eposch based on training the easy dataset
    plt.plot(history_E.history['loss'],'r--')
    plt.plot(history_E.history['val_loss'], 'g--')
    plt.title('Easy model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()    
    
    #plot the change of the loss for each eposch based on training the hard dataset
    plt.plot(history_H.history['loss'],'r--')
    plt.plot(history_H.history['val_loss'], 'g--')
    plt.title('Hard model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()   
     
#------------------------------------------------------------------------------        
#------------------------------------------------------------------------------        

if __name__=='__main__':
    
    epochs = 3
    #exp1()
    #exp2()
    exp3()

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               CODE CEMETARY        
    
