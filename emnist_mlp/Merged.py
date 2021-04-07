import numpy as np
import idx2numpy
from matplotlib import pyplot as plt
from keras import layers
from keras import Sequential
import random
from sklearn.model_selection import train_test_split as Split
from sklearn.metrics import confusion_matrix
###############################################################################
#Import train data for merged
x_train_mer = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-bymerge-train-images-idx3-ubyte')
y_train_mer = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-bymerge-train-labels-idx1-ubyte') 
#Import test data for merged
x_test_mer = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-bymerge-test-images-idx3-ubyte')
y_test_mer = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-bymerge-test-labels-idx1-ubyte')

label_train_mer = np.loadtxt('EMNIST_Raw_Data/emnist-bymerge-mapping.txt' ,delimiter=" ")
print(x_train_mer.shape, y_train_mer.shape) #Get shape of data
print(x_test_mer.shape, y_test_mer.shape) 

#Normalise images to [1,0]
x_train_mer = x_train_mer/255.0
x_test_mer = x_test_mer/255.0

#Transpose for train and test
for i in range(697932):
    x_train_mer[i]=np.transpose(x_train_mer[i])

for i in range(116323):
    x_test_mer[i]=np.transpose(x_test_mer[i])


###############################################################################

def modelbuilding_function(layers_no, input_nodes, output_nodes, activ_func, 
                           activ_out_func, optim = "adam", 
                           lossfunc='sparse_categorical_crossentropy', 
                           met=['accuracy']):

    model = Sequential()
    model.add(layers.Flatten(input_shape=(28,28)))
    model.add(layers.Dropout(0.2))
    for i in range(layers_no):
        model.add(layers.Dense(input_nodes, activation=activ_func))
    model.add(layers.Dense(output_nodes, activation = activ_out_func))
    model.compile(optimizer=optim,
                  loss=lossfunc,
                  metrics=met)
    return model

###############################################################################
#Determine best model parameters
indep = x_train_mer
depen = y_train_mer

x_training, x_val, y_training, y_val = Split(indep, depen, test_size = 0.1)

epochs = [10,20]
batches = [128,256,512]

print("This might take time to run!")
score = []
bestScore = 0
for i in batches:
    for j in epochs:
        random.seed(123)
        model_mer = modelbuilding_function(layers_no=3, input_nodes=256,
                                       output_nodes = 47, activ_func='relu',
                                       activ_out_func='softmax',
                                       lossfunc='sparse_categorical_crossentropy',
                                       met=['accuracy'])
        model_mer.fit(x_training, y_training, epochs = j, batch_size= i, verbose=0)
        scores = model_mer.evaluate(x_test_mer, y_test_mer, verbose=0)
        score.append(scores[1]*100);
        
        if scores[1]>bestScore:
            bestScore=scores[1]
            print("%.2f%% %10i %10i" % (bestScore*100,i,j))
print("\nall done")            
###############################################################################
model_mer.fit(x_train_mer, y_train_mer, epochs= 10, batch_size = 128)
acc_mer = model_mer.evaluate(x_test_mer, y_test_mer, verbose=0)
print("Batch Size = 128 , Epochs = 10 for best model accuracy")
print("Merged Model Accuracy: %.2f%%"%(acc_mer[1]*100))
model_mer.summary()
###############################################################################
x_pred_mer = np.argmax(model_mer.predict(x_test_mer),axis=1)

for i in range(5):
    print("\nThe following merit is a ",x_pred_mer[i])
    plt.imshow(x_test_mer[i])
    plt.show()
###############################################################################
#Validation Data
x_val_mer = x_train_mer[:116323]
partial_x_train_mer = x_train_mer[116323:]
y_val_mer = y_train_mer[:116323]
partial_y_train_mer = y_train_mer[116323:]

print("This might take time to run!")
for i in range(1):  
    x_train_mer,x_val_mer,y_train_mer,y_val_mer = Split(partial_x_train_mer,
                                                        partial_y_train_mer,test_size=0.9)    
    model_mer.fit(x_train_mer, y_train_mer, epochs=10, batch_size=128,verbose=0)
    scores = model_mer.evaluate(x_val_mer, y_val_mer,verbose=0)
    score.append(scores[1]*100);
    
print("Average accuracy across independently trained models : %.2f%%"%np.average(score))
print("Std Dev accuracy across independently trained models : %.2f%%"%np.std(score))

plt.plot(score)
plt.title('Accuracy across all models')
plt.grid()
plt.show


history = model_mer.fit(partial_x_train_mer,partial_y_train_mer,epochs=10,
                         batch_size=128,validation_data=(x_val_mer, y_val_mer))


#Save the training history
hist_dict_mer = history.history
acc_mer = history.history['acc']
val_acc_mer = history.history['val_acc']
loss_values_mer = hist_dict_mer['loss']
val_loss_values_mer = hist_dict_mer['val_loss']
epochs_mer = range(1, len(acc_mer) + 1)

max_mer = max(val_acc_mer)
print(max_mer)
epoch_acc_mer = val_acc_mer.index(max_mer)
print(epoch_acc_mer)
dev_val_mer=np.std(val_acc_mer)
print("Best Validation Accuracy", max_mer, "at Epoch", epoch_acc_mer,
      "Deviation", dev_val_mer )

###############################################################################
# Plot the training and validation losses
plt.plot(epochs_mer, loss_values_mer, 'bo', label='Training loss')
plt.plot(epochs_mer, val_loss_values_mer, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(ymin=0, ymax=0.8)
plt.legend()
plt.show()

#Plot the training and validation accuracies

plt.clf()
acc_values = hist_dict_mer['acc']
val_acc_values = hist_dict_mer['val_acc']
plt.plot(epochs_mer, acc_mer, 'bo', label='Training acc')
plt.plot(epochs_mer, val_acc_mer, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.ylim(ymin=0,ymax=1)
plt.show()
###############################################################################

import pandas as pd
y_pred_mer= model_mer.predict(x_test_mer)
y_pred_mer = np.argmax(y_pred_mer, axis = 1)
print(y_pred_mer)
cm_mer = confusion_matrix(y_test_mer, y_pred_mer)
print("Confusion matrix:\n%s" % cm_mer)
my_mer = pd.DataFrame(cm_mer) 
my_mer.to_csv('matmer.csv', index=False)

from sklearn.metrics import f1_score
f1_score(y_test_mer, y_pred_mer, average='micro')



