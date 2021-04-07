import numpy as np
import idx2numpy
from matplotlib import pyplot as plt
from keras import layers
from keras import Sequential
import random
from sklearn.model_selection import train_test_split as Split
from sklearn.metrics import confusion_matrix
###############################################################################
#Import train data for letters
x_train_let = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-letters-train-images-idx3-ubyte')
y_train_let = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-letters-train-labels-idx1-ubyte') 
label_train_let = np.loadtxt('EMNIST_Raw_Data/emnist-letters-mapping.txt' ,delimiter=" ")
#Import test data for letters
x_test_let = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-letters-test-images-idx3-ubyte')
y_test_let = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-letters-test-labels-idx1-ubyte')

print(x_train_let.shape, y_train_let.shape)
print(x_test_let.shape, y_test_let.shape)

#Normalise images to [1,0]
x_train_let = x_train_let/255.0 
x_test_let = x_test_let/255.0

#Transpose Images
for i in range(124800):
    x_train_let[i]=np.transpose(x_train_let[i])

for i in range(20800):
    x_test_let[i]=np.transpose(x_test_let[i])

###############################################################################
#Build Model
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
indep_let = x_train_let
depen_let = y_train_let

x_training, x_val, y_training, y_val = Split(indep_let, depen_let, test_size = 0.1)

epochs = [10,20]
batches = [128,256,512]

print("This might take time to run!")
score = []
bestScore = 0
for i in batches:
    for j in epochs:
        random.seed(123)
        model_let = modelbuilding_function(layers_no=3, input_nodes=256,
                                       output_nodes = 37, activ_func='relu',
                                       activ_out_func='softmax',
                                       lossfunc='sparse_categorical_crossentropy',
                                       met=['accuracy'])
        model_let.fit(x_training, y_training, epochs = j, batch_size= i, verbose=0)
        scores = model_let.evaluate(x_test_let, y_test_let, verbose=0)
        score.append(scores[1]*100);
        
        if scores[1]>bestScore:
            bestScore=scores[1]
            print("%.2f%% %10i %10i" % (bestScore*100,i,j))
print("\nall done")            
###############################################################################
#Fit best model and test accuracy
model_let.fit(x_train_let, y_train_let, epochs=20, batch_size = 512)
acc_let = model_let.evaluate(x_test_let, y_test_let, verbose=0)
print("Batch Size = 256, Epochs = 20 for best model accuracy")
print("letters Model Accuracy: %.2f%%"%(acc_let[1]*100))
print("Model Summary")
model_let.summary()
###############################################################################
#Validation Data
x_val_let = x_train_let[:20800]
partial_x_train_let = x_train_let[20800:]
y_val_let = y_train_let[:20800] 
partial_y_train_let = y_train_let[20800:]


for i in range(10):  
    x_train_let,x_val_let,y_train_let,y_val_let = Split(partial_x_train_let,
                                                        partial_y_train_let,test_size=0.9)    
    model_let.fit(x_train_let, y_train_let, epochs=20, batch_size=512,verbose=0)
    scores = model_let.evaluate(x_val_let, y_val_let,verbose=0)
    score.append(scores[1]*100);
    
print("Average accuracy across independantly trained models : %.2f%%"%np.average(score))
print("Std Dev accuracy across independantly trained models : %.2f%%"%np.std(score))

plt.plot(score)
plt.title('Accuracy across all models')
plt.grid()
plt.show


history = model_let.fit(partial_x_train_let,partial_y_train_let,epochs=20,
                        batch_size=256,validation_data=(x_val_let, y_val_let))

#Save the training history
hist_dict_let = history.history
acc_let = history.history['acc']
val_acc_let = history.history['val_acc']
loss_values_let = hist_dict_let['loss']
val_loss_values_let = hist_dict_let['val_loss']
epochs_let = range(1, len(acc_let) + 1)

max_let = max(val_acc_let)
print(max_let)
epoch_acc_let = val_acc_let.index(max_let)
print(epoch_acc_let)
dev_val_let=np.std(val_acc_let)
print("Best Validation Accuracy", max_let, "at Epoch", epoch_acc_let,
      "Deviation", dev_val_let )

###############################################################################
# Plot the training and validation losses
plt.plot(epochs_let, loss_values_let, 'bo', label='Training loss')
plt.plot(epochs_let, val_loss_values_let, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(ymin=0, ymax=0.8)
plt.legend()
plt.show()

#Plot the training and validation accuracies

plt.clf()
acc_values = hist_dict_let['acc']
val_acc_values = hist_dict_let['val_acc']
plt.plot(epochs_let, acc_let, 'bo', label='Training acc')
plt.plot(epochs_let, val_acc_let, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.ylim(ymin=0,ymax=1)
plt.show()
###############################################################################

import pandas as pd
y_pred_let= model_let.predict(x_test_let)
y_pred_let = np.argmax(y_pred_let, axis = 1)
print(y_pred_let)
cm_let = confusion_matrix(y_test_let, y_pred_let)
print("Confusion matrix:\n%s" % cm_let)
my_let = pd.DataFrame(cm_let) 
my_let.to_csv('matlet.csv', index=False)

from sklearn.metrics import f1_score
f1_score(y_test_let, y_pred_let, average='micro')









    
