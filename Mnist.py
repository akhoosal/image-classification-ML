import numpy as np
import idx2numpy
from matplotlib import pyplot as plt
from keras import layers
from keras import Sequential
import random
from sklearn.model_selection import train_test_split as Split
from sklearn.metrics import confusion_matrix

###############################################################################
#Import train data for mnist
x_train_mni = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-mnist-train-images-idx3-ubyte')
y_train_mni = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-mnist-train-labels-idx1-ubyte') 
label_train_mni = np.loadtxt('EMNIST_Raw_Data/emnist-mnist-mapping.txt' ,delimiter=" ")
#Import test data for mnist
x_test_mni = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-mnist-test-images-idx3-ubyte')
y_test_mni = idx2numpy.convert_from_file('EMNIST_Raw_Data/emnist-mnist-test-labels-idx1-ubyte')

print(x_train_mni.shape, y_train_mni.shape)
print(x_test_mni.shape, y_test_mni.shape)

#Normalise images to [1,0]
x_train_mni = x_train_mni/255.0 
x_test_mni = x_test_mni/255.0

#Transpose Images
for i in range(60000):
    x_train_mni[i]=np.transpose(x_train_mni[i])

for i in range(10000):
    x_test_mni[i]=np.transpose(x_test_mni[i])

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
indep_mni = x_train_mni
depen_mni = y_train_mni

x_training, x_val, y_training, y_val = Split(indep_mni, depen_mni, test_size = 0.1)

epochs = [10,20]
batches = [128,256,512]

print("This might take time to run!")
score = []
bestScore = 0
for i in batches:
    for j in epochs:
        random.seed(123)
        model_mni = modelbuilding_function(layers_no=3, input_nodes=256,
                                       output_nodes = 10, activ_func='relu',
                                       activ_out_func='softmax',
                                       lossfunc='sparse_categorical_crossentropy',
                                       met=['accuracy'])
        model_mni.fit(x_training, y_training, epochs = j, batch_size= i, verbose=0)
        scores = model_mni.evaluate(x_test_mni, y_test_mni, verbose=0)
        score.append(scores[1]*100);
        
        if scores[1]>bestScore:
            bestScore=scores[1]
            print("%.2f%% %10i %10i" % (bestScore*100,i,j))
print("\nall done")            
###############################################################################
#Fit best model and test accuracy
model_mni.fit(x_train_mni, y_train_mni, epochs=20, batch_size = 256)
acc_mni = model_mni.evaluate(x_test_mni, y_test_mni, verbose=0)
print("Batch Size = 256, Epochs = 20 for best model accuracy")
print("mnist Model Accuracy: %.2f%%"%(acc_mni[1]*100))
print("Model Summary")
model_mni.summary()
###############################################################################
x_pred_mni = np.argmax(model_mni.predict(x_test_mni),axis=1)
for i in range(5):
    print("\nThe following mniit is a ",x_pred_mni[i])
    plt.imshow(x_test_mni[i])
    plt.show()
###############################################################################
#Validation Data
x_val_mni = x_train_mni[:10000]
partial_x_train_mni = x_train_mni[10000:]
y_val_mni = y_train_mni[:10000] 
partial_y_train_mni = y_train_mni[10000:]


for i in range(10):  
    x_train_mni,x_val_mni,y_train_mni,y_val_mni = Split(partial_x_train_mni,
                                                        partial_y_train_mni,test_size=0.9)    
    model_mni.fit(x_train_mni, y_train_mni, epochs=10, batch_size=256,verbose=0)
    scores = model_mni.evaluate(x_val_mni, y_val_mni,verbose=0)
    score.append(scores[1]*100);
    
print("Average accuracy across independantly trained models : %.2f%%"%np.average(score))
print("Std Dev accuracy across independantly trained models : %.2f%%"%np.std(score))

plt.plot(score)
plt.title('Accuracy across all models')
plt.grid()
plt.show


history = model_mni.fit(partial_x_train_mni,partial_y_train_mni,epochs=20,
                        batch_size=256,validation_data=(x_val_mni, y_val_mni))

#Save the training history
hist_dict_mni = history.history
acc_mni = history.history['acc']
val_acc_mni = history.history['val_acc']
loss_values_mni = hist_dict_mni['loss']
val_loss_values_mni = hist_dict_mni['val_loss']
epochs_mni = range(1, len(acc_mni) + 1)

max_mni = max(val_acc_mni)
print(max_mni)
epoch_acc_mni = val_acc_mni.index(max_mni)
print(epoch_acc_mni)
dev_val_mni=np.std(val_acc_mni)
print("Best Validation Accuracy", max_mni, "at Epoch", epoch_acc_mni,
      "Deviation", dev_val_mni )

###############################################################################
# Plot the training and validation losses
plt.plot(epochs_mni, loss_values_mni, 'bo', label='Training loss')
plt.plot(epochs_mni, val_loss_values_mni, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(ymin=0, ymax=0.2)
plt.legend()
plt.show()

#Plot the training and validation accuracies

plt.clf()
acc_values = hist_dict_mni['acc']
val_acc_values = hist_dict_mni['val_acc']
plt.plot(epochs_mni, acc_mni, 'bo', label='Training acc')
plt.plot(epochs_mni, val_acc_mni, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.ylim(ymin=0,ymax=1.5)
plt.show()
###############################################################################



import pandas as pd
y_pred_mni= model_mni.predict(x_test_mni)
y_pred_mni = np.argmax(y_pred_mni, axis = 1)
print(y_pred_mni)
cm_mni = confusion_matrix(y_test_mni, y_pred_mni)
print("Confusion matrix:\n%s" % cm_mni)
my_mni = pd.DataFrame(cm_mni) 
my_mni.to_csv('matmni.csv', index=False)


from sklearn.metrics import f1_score
f1_score(y_test_mni, y_pred_mni, average='micro')










    