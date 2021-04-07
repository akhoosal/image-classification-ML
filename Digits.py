import numpy as np
import idx2numpy
from matplotlib import pyplot as plt
from keras import layers
from keras import Sequential
import random
from sklearn.model_selection import train_test_split as Split
from sklearn.metrics import confusion_matrix
###############################################################################
#Import train data for digits
x_train_dig = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-digits-train-images-idx3-ubyte')
y_train_dig = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-digits-train-labels-idx1-ubyte') 
#Import test data for digits
x_test_dig = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-digits-test-images-idx3-ubyte')
y_test_dig = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-digits-test-labels-idx1-ubyte')

label_train_dig = np.loadtxt('EMNIST_Raw_Data/emnist-digits-mapping.txt' ,delimiter=" ")
print(x_train_dig.shape, y_train_dig.shape) #Get shape of data
print(x_test_dig.shape, y_test_dig.shape) 

#Normalise images to [1,0]
x_train_dig = x_train_dig/255.0
x_test_dig = x_test_dig/255.0

#Transpose for train and test
for i in range(240000):
    x_train_dig[i]=np.transpose(x_train_dig[i])

for i in range(40000):
    x_test_dig[i]=np.transpose(x_test_dig[i])

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
indep = x_train_dig
depen = y_train_dig

x_training, x_val, y_training, y_val = Split(indep, depen, test_size = 0.1)

epochs = [10,20]
batches = [128,256,512]

print("This might take time to run!")
score = []
bestScore = 0
for i in batches:
    for j in epochs:
        random.seed(123)
        model_dig = modelbuilding_function(layers_no=3, input_nodes=256,
                                       output_nodes = 10, activ_func='relu',
                                       activ_out_func='softmax',
                                       lossfunc='sparse_categorical_crossentropy',
                                       met=['accuracy'])
        model_dig.fit(x_training, y_training, epochs = j, batch_size= i, verbose=0)
        scores = model_dig.evaluate(x_test_dig, y_test_dig, verbose=0)
        score.append(scores[1]*100);
        
        if scores[1]>bestScore:
            bestScore=scores[1]
            print("%.2f%% %10i %10i" % (bestScore*100,i,j))
print("\nall done")            
###############################################################################
model_dig.fit(x_train_dig, y_train_dig, epochs=10, batch_size = 128)
acc_dig = model_dig.evaluate(x_test_dig, y_test_dig, verbose=0)
print("Batch Size = 128 , Epochs = 10 for best model accuracy")
print("Digits Model Accuracy: %.2f%%"%(acc_dig[1]*100))
model_dig.summary()
###############################################################################
x_pred_dig = np.argmax(model_dig.predict(x_test_dig),axis=1)

for i in range(5):
    print("\nThe following digit is a ",x_pred_dig[i])
    plt.imshow(x_test_dig[i])
    plt.show()
###############################################################################
#Validation Data
random.seed(123)
x_val_dig = x_train_dig[:40000]
partial_x_train_dig = x_train_dig[40000:]
y_val_dig = y_train_dig[:40000]
partial_y_train_dig = y_train_dig[40000:]

print("This might take time to run!")
for i in range(10):  
    x_train_dig,x_val_dig,y_train_dig,y_val_dig = Split(partial_x_train_dig,
                                                        partial_y_train_dig,test_size=0.9)    
    model_dig.fit(x_train_dig, y_train_dig, epochs=10, batch_size=128,verbose=0)
    scores = model_dig.evaluate(x_val_dig, y_val_dig,verbose=0)
    score.append(scores[1]*100);
    
print("Average accuracy across independently trained models : %.2f%%"%np.average(score))
print("Std Dev accuracy across independently trained models : %.2f%%"%np.std(score))

plt.plot(score)
plt.title('Accuracy across all models')
plt.grid()
plt.show

history = model_dig.fit(partial_x_train_dig,partial_y_train_dig,epochs=20,
                        batch_size=128,validation_data=(x_val_dig, y_val_dig))

#Save the training history
hist_dict_dig = history.history
acc_dig = history.history['acc']
val_acc_dig = history.history['val_acc']
loss_values_dig = hist_dict_dig['loss']
val_loss_values_dig = hist_dict_dig['val_loss']
epochs_dig = range(1, len(acc_dig) + 1)

max_dig = max(val_acc_dig)
print(max_dig)
epoch_acc_dig = val_acc_dig.index(max_dig)
print(epoch_acc_dig)
dev_val_dig=np.std(val_acc_dig)
print("Best Validation Accuracy", max_dig, "at Epoch", epoch_acc_dig,
      "Deviation", dev_val_dig )


###############################################################################
# Plot the training and validation losses
plt.plot(epochs_dig, loss_values_dig, 'bo', label='Training loss')
plt.plot(epochs_dig, val_loss_values_dig, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(ymin=0, ymax=0.8)
plt.legend()
plt.show()

#Plot the training and validation accuracies

plt.clf()
acc_values = hist_dict_dig['acc']
val_acc_values = hist_dict_dig['val_acc']
plt.plot(epochs_dig, acc_dig, 'bo', label='Training acc')
plt.plot(epochs_dig, val_acc_dig, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.ylim(ymin=0,ymax=1)
plt.show()
###############################################################################
import pandas as pd
y_pred_dig= model_dig.predict(x_test_dig)
y_pred_dig = np.argmax(y_pred_dig, axis = 1)
print(y_pred_dig)
cm_dig = confusion_matrix(y_test_dig, y_pred_dig)
print("Confusion matrix:\n%s" % cm_dig)
my_dig = pd.DataFrame(cm_dig) 
my_dig.to_csv('matdig.csv', index=False)

from sklearn.metrics import f1_score
f1_score(y_test_dig, y_pred_dig, average='micro')





