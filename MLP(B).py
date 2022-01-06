import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape to 28 x 28 pixels = 784 features
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Convert into greyscale
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Convert target classes （0-9）to categorical ones； one-hot encode
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# second method: normalize the train&test numpy array
# x_train = keras.utils.normalize(x_train,axis=1)
# x_test = keras.utils.normalize(x_test,axis=1)

num_hid = [3, 4]
num_nodes = [32, 64]
Act_func = [tf.nn.sigmoid, tf.nn.relu]

# hyper_paras matrix
hyper_paras = ([[num_hid[0], num_nodes[0], Act_func[0]],
        [num_hid[0], num_nodes[0], Act_func[1]],
        [num_hid[0], num_nodes[1], Act_func[0]],
        [num_hid[0], num_nodes[1], Act_func[1]],
        [num_hid[1], num_nodes[0], Act_func[0]],
        [num_hid[1], num_nodes[0], Act_func[1]],
        [num_hid[1], num_nodes[1], Act_func[0]],
        [num_hid[1], num_nodes[1], Act_func[1]]])

plt.figure(figsize=(20, 10), dpi=100)

# grid search loop
for i in range(len(hyper_paras)):

    model = Sequential()
    if i <= num_hid[0]:
        counter = 1
        while counter <= num_hid[0]:
            model.add(keras.layers.Dense(hyper_paras[counter][1], activation=hyper_paras[counter][2]))
            counter = counter + 1
    elif i >= num_hid[1]:
        counter = 1
        while counter <= num_hid[1]:
            model.add(keras.layers.Dense(hyper_paras[counter][1], activation=hyper_paras[counter][2]))
            counter = counter + 1

    print('hyper_paras: ', hyper_paras[i])
    model.add(keras.layers.Dense(10, activation=hyper_paras[counter][2]))
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                # change to sparse_categorical_crossentropy if choose the normalize way
                metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, validation_split=0.1, batch_size=200)
    model.summary()

    val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=128)
    print(val_loss, val_acc)

    plt.plot(history.history['acc'], label='acc=%d'% i)
    plt.plot(history.history['val_acc'], label='val_acc=%d'% i)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.4, linestyle=':')
plt.show()


# best_hyper_para is [3,32,sigmoid], the top orange line
# the numbers of hidden layer do the dominate role in this model