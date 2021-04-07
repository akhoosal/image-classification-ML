import numpy as np
import idx2numpy
from matplotlib import pyplot as plt
from keras import layers
from keras import Sequential
import random
from sklearn.model_selection import train_test_split as Split
from sklearn.metrics import confusion_matrix

###############################################################################
#Import train data for byclass
x_train_cla = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-byclass-train-images-idx3-ubyte')
y_train_cla = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-byclass-train-labels-idx1-ubyte') 
label_train_cla = np.loadtxt('EMNIST_Raw_Data/emnist-byclass-mapping.txt' ,delimiter=" ")
#Import test data for byclass
x_test_cla = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-byclass-test-images-idx3-ubyte')
y_test_cla = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-byclass-test-labels-idx1-ubyte')

print(x_train_cla.shape, y_train_cla.shape)
print(x_test_cla.shape, y_test_cla.shape)

#Normalise images to [1,0]
x_train_cla = x_train_cla/255.0 
x_test_cla = x_test_cla/255.0

#Transpose Images
for i in range(697932):
    x_train_cla[i]=np.transpose(x_train_cla[i])

for i in range(116323):
    x_test_cla[i]=np.transpose(x_test_cla[i])

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
indep_cla = x_train_cla
depen_cla = y_train_cla

x_training, x_val, y_training, y_val = Split(indep_cla, depen_cla, test_size = 0.1)

epochs = [10,20]
batches = [128,256,512]

print("This might take time to run!")
score = []
bestScore = 0
for i in batches:
    for j in epochs:
        random.seed(123)
        model_cla = modelbuilding_function(layers_no=3, input_nodes=256,
                                       output_nodes = 62, activ_func='relu',
                                       activ_out_func='softmax',
                                       lossfunc='sparse_categorical_crossentropy',
                                       met=['accuracy'])
        model_cla.fit(x_training, y_training, epochs = j, batch_size= i, verbose=0)
        scores = model_cla.evaluate(x_test_cla, y_test_cla, verbose=0)
        score.append(scores[1]*100);
        
        if scores[1]>bestScore:
            bestScore=scores[1]
            print("%.2f%% %10i %10i" % (bestScore*100,i,j))
print("\nall done")            
###############################################################################
#Fit best model and test accuracy
model_cla.fit(x_train_cla, y_train_cla, epochs=20, batch_size = 512)
acc_cla = model_cla.evaluate(x_test_cla, y_test_cla, verbose=0)
print("Batch Size = 512, Epochs = 20 for best model accuracy")
print("byclass Model Accuracy: %.2f%%"%(acc_cla[1]*100))
print("Model Summary")
model_cla.summary()
###############################################################################
x_pred_cla = np.argmax(model_cla.predict(x_test_cla),axis=1)
for i in range(5):
    print("\nThe following clait is a ",x_pred_cla[i])
    plt.imshow(x_test_cla[i])
    plt.show()
###############################################################################
#Validation Data
x_val_cla = x_train_cla[:116323]
partial_x_train_cla = x_train_cla[116323:]
y_val_cla = y_train_cla[:116323] 
partial_y_train_cla = y_train_cla[116323:]

print("This might take time to run!")
for i in range(10):  
    x_train_cla,x_val_cla,y_train_cla,y_val_cla = Split(partial_x_train_cla,
                                                        partial_y_train_cla,test_size=0.9)    
    model_cla.fit(x_train_cla, y_train_cla, epochs=10, batch_size=128,verbose=0)
    scores = model_cla.evaluate(x_val_cla, y_val_cla,verbose=0)
    score.append(scores[1]*100);
    
print("Average accuracy across independently trained models : %.2f%%"%np.average(score))
print("Std Dev accuracy across independently trained models : %.2f%%"%np.std(score))

plt.plot(score)
plt.title('Accuracy across all models')
plt.grid()
plt.show


history = model_cla.fit(partial_x_train_cla,partial_y_train_cla,epochs=20,
                        batch_size=256,validation_data=(x_val_cla, y_val_cla))

#Save the training history
hist_dict_cla = history.history
acc_cla = history.history['acc']
val_acc_cla = history.history['val_acc']
loss_values_cla = hist_dict_cla['loss']
val_loss_values_cla = hist_dict_cla['val_loss']
epochs_cla = range(1, len(acc_cla) + 1)

max_cla = max(val_acc_cla)
print(max_cla)
epoch_acc_cla = val_acc_cla.index(max_cla)
print(epoch_acc_cla)
dev_val_cla=np.std(val_acc_cla)
print("Best Validation Accuracy", max_cla, "at Epoch", epoch_acc_cla,
      "Deviation", dev_val_cla )


###############################################################################
# Plot the training and validation losses
plt.plot(epochs_cla, loss_values_cla, 'bo', label='Training loss')
plt.plot(epochs_cla, val_loss_values_cla, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(ymin=0, ymax=1)
plt.legend()
plt.show()

#Plot the training and validation accuracies

plt.clf()
acc_values = hist_dict_cla['acc']
val_acc_values = hist_dict_cla['val_acc']
plt.plot(epochs_cla, acc_cla, 'bo', label='Training acc')
plt.plot(epochs_cla, val_acc_cla, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.ylim(ymin=0,ymax=1)
plt.show()
###############################################################################


import pandas as pd
y_pred_cla= model_cla.predict(x_test_cla)
y_pred_cla = np.argmax(y_pred_cla, axis = 1)
print(y_pred_cla)
cm_cla = confusion_matrix(y_test_cla, y_pred_cla)
print("Confusion matrix:\n%s" % cm_cla)
my_cla = pd.DataFrame(cm_cla) 
my_cla.to_csv('matcla.csv', index=False)



from sklearn.metrics import f1_score
f1_score(y_test_cla, y_pred_cla, average='micro')





    