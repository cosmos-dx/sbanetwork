
# **SBANetwork**

A CPU based lightweight neural network library inspired by keras.


```from sbanetwork.core.model import Model```

## **Model Class**

Model Class contains methods 

**INIT** 
```
model = Model([
    Dense(128, activation="relu"), 
    Dropout(rate=0.3, name='dropout_1'),
    Dense(64, activation="relu", name='dense_2'),
    Dropout(rate=0.3, name='dropout_2'),
    Dense(num_classes, activation="softmax", name='output')
])
```
Dense is inspired by Keras. It is used to create the layers. The first dense layer can automatically detect the input shape.

```
Dense(128, activation="relu"), #128 units in Neural Network 
```
Here in Dense you can also use Regulizers which are also used to reduce the complexity and to reduce the overfitting.

There are given **Regulizers** -

    1. l1
    2. l2
    3. maxnorm

Dropout is used to prevent the Overfitting into a model. Where you can drop the random nodes in every epoch so that it can identify the pattern instead of remembering.
```
Dropout(rate=0.3, name='dropout_1'), #rate(percentage),  name(dropout name)
```
There are lots of **activation functions** are being used - 

    1. relu
    2. leaky_relu
    3. elu
    4. selu
    5. swish
    6. mish
    7. sigmoid
    8. tanh
    9. linear
    10. softmax
    11. gelu
    12. bent_identity
    13. gaussian
    14. softplus

**Compile**

compile(optimizer='sgd', loss='mse', learning_rate=0.01, metrics=None)

There are lots of **optimizers** which are -

    1. sgd 
    2. sgdmomentum
    3. nag
    4. adagrad
    5. rmsprop
    6. adam
The **losses** we are using are -
    
    1. mse
    2. mae
    3. binary_crossentropy, bce
    4. categorical_crossentropy, ce
    5. huber

```
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    learning_rate=0.001,
    metrics=['accuracy']
)
```
**Fit**

Fit method is used to train the model with multiple parameters given in different environments.

fit(x_train, y_train, epochs=1, batch_size=32, validation_data=None, verbose=True, callbacks_list=None):
It returns the training history. 

The **callbacks** we are using -

    1. early_stopping
    2. model_checkpoints
    3. learning_rate_scheduler
```
history = model.fit(
    X, y,
    epochs=50,
    batch_size=16,
    validation_data=(X, y),
    verbose=True
)
```
**Summary**

model.summary()

**Predict**

predict(x, batch_size=32):

**Export**

model.export(path)

**Load**

model.load(path)

## **Pendings**

    1. To apply callbacks and Regulizers
    2. Conv2d, flattening, Pooling etc
    3. Validation Split
    4. Preprocess 
    5. Experimental functions.


