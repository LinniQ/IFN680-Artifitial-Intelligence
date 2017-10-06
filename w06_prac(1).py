
import numpy as np
from tensorflow.contrib import keras 
import matplotlib.pyplot as plt
import assign2_utils
def ex1():
    '''
    Save the arrays x_train, y_train, x_test and y_test
    into a single npz file named 'mnist_dataset.npz'   
    '''
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # INSERT YOUR CODE HERE
    np.savez('mnist_dataset.npz', x_train= x_train, y_train=y_train, x_test=x_test, y_test=y_test)
                             
def ex2():
    '''
    Read back the arrays x_train, y_train, x_test and y_test
    from the npz file named 'mnist_dataset.npz'.
    Then, print the shape and dtype of these numpy arrays.
    
    '''
    with np.load('mnist_dataset.npz') as npzfile:
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']    # INSERT YOUR CODE HERE
        x_test  = npzfile['x_test']   # INSERT YOUR CODE HERE
        y_test  = npzfile['y_test']   # INSERT YOUR CODE HERE

    
    print('x_train : ', x_train.shape, x_train.dtype)
    print('y_train : ', y_train.shape, y_train.dtype)
    print('x_test : ', x_test.shape, x_test.dtype)
    print('y_test : ', y_test.shape, y_test.dtype)


def ex3():
    '''
    Read back the arrays x_train, y_train, x_test and y_test
    from the npz file named 'mnist_dataset.npz'.
    Then, print the shape and dtype of these numpy arrays.
    
    '''
    with np.load('mnist_dataset.npz') as npzfile:   # INSERT YOUR CODE HERE
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']    # INSERT YOUR CODE HERE
        x_test  = npzfile['x_test']   # INSERT YOUR CODE HERE
        y_test  = npzfile['y_test'] 
        X_train = assign2_utils.random_deform(x_train[2],45,0.3)
        for i in range(30,35):
            plt.imshow(x_train[2],cmap='rainbow')
            #plt.imshow(X_train,cmap='gray')     # INSERT YOUR CODE HERE     
            plt.title(str(y_train[2]))  # INSERT YOUR CODE HERE
            plt.show()
        for i in range(30,35):
            #plt.imshow(x_train[2],cmap='rainbow')
            plt.imshow(X_train,cmap='gray')     # INSERT YOUR CODE HERE     
            plt.title(str(y_train[2]))  # INSERT YOUR CODE HERE
            plt.show()

def ex4():
    '''
    Build, train and evaluate a CNN on the mnist dataset
    
    '''
    # hyperparameters
    batch_size = 128
    epochs = 1
    input_shape = (28,28,1)
    with np.load('mnist_dataset.npz') as npzfile:
        # INSERT YOUR CODE HERE
        #data
        x_train = npzfile['x_train']
        y_train = npzfile['y_train']    # INSERT YOUR CODE HERE
        x_test  = npzfile['x_test']   # INSERT YOUR CODE HERE
        y_test  = npzfile['y_test'] 
        x_train = x_train.reshape(x_train.shape[0], *input_shape).astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0], *input_shape).astype(np.float32)
        
        num_classes = len(np.unique(y_train))
        
        #preprocessing
        x_train = x_train / 255
        x_test = x_test / 255
        X_train = np.zeros_like(x_train)
        X_test = np.zeros_like(x_test)
        for i in range(28):
            X_train[i] = assign2_utils.random_deform(x_train[i],45,0.3)
            X_test[i] = assign2_utils.random_deform(x_test[i],45,0.3)

        # cast to one_hot vector
        y_train = keras.utils.to_categorical(y_train, num_classes)
        print(y_train)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        '''
        Y_train = np.zeros_like(y_train)
        Y_test = np.zeros_like(y_test)
        for i in range(28):
            Y_train[i] = assign2_utils.random_deform(y_train[i],45,0.3)
            Y_test[i] = assign2_utils.random_deform(y_test[i],45,0.3)
        '''
        #build graph
        model = keras.models.Sequential()  #!!!arg keras.models.Sequential()
        
        model.add(keras.layers.Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=input_shape))
        model.add(keras.layers.Conv2D(64, kernel_size=(3,3),activation='relu'))
        model.add(keras.layers.MaxPooling2D((2,2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128,activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10,activation='softmax'))
        
        
        
        model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                      metrics=['accuracy'])
        
        #train
        model.fit(X_train,y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(X_test,y_test))
        '''
        model.fit(x_train,y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test,y_test))
        '''
        '''
        img_rows, img_cols = x_train.shape[1:3]
        num_classes = len(np.unique(y_train))
        
        # reshape the input arrays to 4D (batch_size, rows, columns, channels)
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
        batch_size = 128
        epochs = 12
    
        #    epochs = 3 # debugging code
        #    x_train = x_train[:8000]
        #    y_train = y_train[:8000]
    
    
        # convert to float32 and rescale between 0 and 1
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        #
        # convert class vectors to binary class matrices (aka "sparse coding" or "one hot")
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = # INSERT YOUR CODE HERE    #
        
        model = keras.models.S# INSERT YOUR CODE HERE 
        
        model.add(keras.layers.Conv2D( # INSERT YOUR CODE HERE
            
        # INSERT YOUR CODE HERE    
    
        model.compile( # INSERT YOUR CODE HERE
        
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        '''          
        score = model.evaluate(X_test, y_test, verbose=0)
        
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        

if __name__ == '__main__':
    ex1()
    ex2()
    ex3()
    ex4()    