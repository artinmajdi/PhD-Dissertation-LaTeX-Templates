import os
import pathlib 
from main.aims.aim1_1_taxonomy.SAM import resnet_cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import cv2

tf.config.run_functions_eagerly(False)


# Reference
# https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/blob/master/zoo/resnet/resnet_cifar10.py
def get_training_model(n_classes=10, activation='softmax'):
    # ResNet20
    n = 2
    depth =  n * 9 + 2
    n_blocks = ((depth - 2) // 9) - 1

    # The input tensor
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    # The Stem Convolution Group
    x = resnet_cifar10.stem(inputs)

    # The learner
    x = resnet_cifar10.learner(x, n_blocks)

    # The Classifier for 10 classes
    outputs = resnet_cifar10.classifier(x=x, n_classes=n_classes, activation=activation)

    # Instantiate the Model
    model = tf.keras.Model(inputs, outputs)

    return model

def plot_history(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

class ArtinStuff():

    def __init__(self, mode='coarse'):
        self.mode = mode

    @staticmethod
    def load_cifar100_dataset(X, Y, strategy, mode='fine', batch_size=128):

        def formatting(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            label = tf.cast(label, tf.int32)
            return image, label

        def scale(image, label):
            image = tf.image.resize(image, [224, 224])
            return image, label

        def augment(image,label):
            image = tf.image.resize_with_crop_or_pad(image, 40, 40) # Add 8 pixels of padding
            image = tf.image.random_crop(image, size=[32, 32, 3]) # Random crop back to 32x32
            image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
            image = tf.clip_by_value(image, 0., 1.)
            return image, label

        BATCH_SIZE = batch_size * strategy.num_replicas_in_sync
        print(f"Batch size: {BATCH_SIZE}")
        AUTO = tf.data.AUTOTUNE


        train_ds = tf.data.Dataset.from_tensor_slices( ( X[mode]['train'] ,  Y[mode]['train']  ) )
        train_ds = (
            train_ds
            .shuffle(1024)
            .map(formatting, num_parallel_calls=AUTO)
            .map(augment, num_parallel_calls=AUTO)
            .map(scale, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )

        test_ds = tf.data.Dataset.from_tensor_slices( ( X[mode]['test'] ,  Y[mode]['test']  ) )
        test_ds = (
            test_ds
            .map(formatting, num_parallel_calls=AUTO)
            .map(scale, num_parallel_calls=AUTO)
            .batch(BATCH_SIZE)
            .prefetch(AUTO)
        )

        return train_ds, test_ds

    @staticmethod
    def load_cifar100_raw_data():

        # Downloading the dataset
        label =  dict(train=None, valid=None, test=None)

        X = { 'coarse':label.copy() , 'fine':label.copy() , 'merged':label.copy()}
        Y = { 'coarse':label.copy() , 'fine':label.copy() , 'merged':label.copy() }

        for label_mode in  ['coarse' , 'fine']:
            ( X[label_mode]['train'], Y[label_mode]['train'] ) , ( X[label_mode]['test'], Y[label_mode]['test'] ) =tf.keras.datasets.cifar100.load_data(label_mode=label_mode)

        # for mode in  ['train' , 'test']:
        #   X['fine'][mode] = cv2.resize(X['fine'][mode] , (224,224))
        #   X['coarse'][mode]  = X['fine'][mode].copy()

        # Merging the fine and coarse labels
        for mode in ['train' , 'test']:
            Y['merged'][mode] = tf.keras.utils.to_categorical( Y['fine'][mode]+20 , 120 ) + tf.keras.utils.to_categorical( Y['coarse'][mode] , 120 )
            X['merged'][mode] = X['fine'][mode].copy()

        # for thresh_technique in ['train', 'test']:
        #     Y['coarse'][thresh_technique] = tf.keras.utils.to_categorical( Y['coarse'][thresh_technique] ,  20 )
        #     Y['fine'][thresh_technique]    = tf.keras.utils.to_categorical( Y['fine'][thresh_technique]   ,  100 )

        return X, Y

    @staticmethod
    def tpu_gpu_initialization():

        try: # detect TPUs
            tpu = None
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu) # experimental

        except ValueError: # detect GPUs
            strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines

        return strategy

        print("Number of accelerators: ", strategy.num_replicas_in_sync)

    @staticmethod
    def train_callbacks():
        train_callbacks = [
            tf.keras.callbacks.EarlyStopping(  monitor="val_loss", patience=10,   restore_best_weights=True ),
            tf.keras.callbacks.ReduceLROnPlateau(  monitor="val_loss", factor=0.5,  patience=3, verbose=1  )
        ]
        return train_callbacks

    @staticmethod
    def weighted_bce_loss():

        def func_loss(y_true,y_pred):

            NUM_CLASSES = y_pred.shape[1]
            loss = 0

            for d in range(NUM_CLASSES):

                y_true = tf.cast(y_true, tf.float32)
                # mask   = tf.keras.backend.cast( tf.keras.backend.not_equal(y_true[:,d], -5), tf.keras.backend.floatx() )
                # loss  += W[d]*tf.keras.losses.binary_crossentropy( y_true[:,d] * mask,  y_pred[:,d] * mask )
                loss += tf.keras.losses.binary_crossentropy( y_true[ : , d ]  ,   y_pred[ : , d ] )

            return tf.divide( loss,  tf.cast(NUM_CLASSES,tf.float32) )

        return func_loss

    @staticmethod
    def fit_SAM( strategy ,  train_ds ,  test_ds , n_classes=10, activation='softmax', loss='sparse_categorical_crossentropy'):

        with strategy.scope():
            model = SAMModel( resnet_model=get_training_model(n_classes=n_classes, activation=activation) )

        if loss == 'weighted_bce_loss':
          loss = ArtinStuff.weighted_bce_loss()

        model.compile( optimizer="adam", loss=loss, metrics=["accuracy"] ) 
        print(f"Total learnable parameters: {model.resnet_model.count_params()/1e6} M")

        start = time.time()
        history = model.fit( train_ds , validation_data=test_ds ,   callbacks=ArtinStuff.train_callbacks() ,  epochs=100 )

        print(f"Total training time: {(time.time() - start)/60.} minutes")

        return history




class SAMModel(tf.keras.Model):

    def __init__(self, resnet_model, rho=0.05):
        """
        p, q = 2 for optimal results as suggested in the paper
        (Section 2)
        """
        super(SAMModel, self).__init__()
        self.resnet_model = resnet_model
        self.rho = rho

    def train_step(self, data):

        (images, labels) = data
        e_ws = []

        with tf.GradientTape() as tape:
            predictions = self.resnet_model(images)
            loss = self.compiled_loss(labels, predictions)

        trainable_params = self.resnet_model.trainable_variables
        gradients = tape.gradient(loss, trainable_params)
        grad_norm = self._grad_norm(gradients)
        scale = self.rho / (grad_norm + 1e-12)

        for (grad, param) in zip(gradients, trainable_params):
            e_w = grad * scale
            param.assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            predictions = self.resnet_model(images)
            loss = self.compiled_loss(labels, predictions)

        sam_gradients = tape.gradient(loss, trainable_params)

        for (param, e_w) in zip(trainable_params, e_ws):
            param.assign_sub(e_w)

        self.optimizer.apply_gradients( zip(sam_gradients, trainable_params) )

        self.compiled_metrics.update_state(labels, predictions)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):

        (images, labels) = data

        predictions = self.resnet_model(images, training=False)
        loss = self.compiled_loss(labels, predictions)

        self.compiled_metrics.update_state(labels, predictions)

        return {m.name: m.result() for m in self.metrics}

    def _grad_norm(self, gradients):
        norm = tf.norm(  tf.stack([ tf.norm(grad) for grad in gradients if grad is not None ])  )
        return norm
