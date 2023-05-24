import argparse
import multiprocessing
import pickle
import typing

import mlflow
import numpy as np
import pandas as pd
import researchpy as rp
import seaborn as sns
import sklearn
import tensorflow as tf
from crowdkit import aggregation as crowdkit_aggregation
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline  # ,BSpline
from scipy.special import bdtrc
from sklearn import ensemble as sk_ensemble
from sklearn import metrics as sk_metrics
from tqdm import tqdm_notebook as tqdm

from . import load_data
from .utils_mlflow import AIM1_3_MLFLOW_SETUP


class Dict2Class:
    """ It takes a dictionary and turns it into a class """

    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def func_callBacks(dir_save='', mode='min', monitor='val_loss'):
    checkPointer = tf.keras.callbacks.ModelCheckpoint(filepath=dir_save, monitor=monitor, verbose=1,
                                                      save_best_only=True, mode=mode)

    # Reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, min_delta=0.005 , patience=10, verbose=1, save_best_only=True, mode=mode , min_lr=0.9e-5 , )

    # CSVLogger = tf.keras.callbacks.CSVLogger(dir_save + '/results.csv', separator=',', append=False)

    # earlyStopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, verbose=1, mode=mode, restore_best_weights=True)

    return [checkPointer]  # [ earlyStopping , CSVLogger]

def reading_terminal_inputs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", help="number of epochs")
    parser.add_argument("--bsize", help="batch size")
    parser.add_argument("--max_sample", help="maximum number of training samples")
    parser.add_argument("--naug", help="number of augmentations")

    """ Xception          VG16                 VGG19           DenseNet201
        ResNet50          ResNet50V2           ResNet101       DenseNet169
        ResNet101V2       ResNet152            ResNet152V2     DenseNet121
        InceptionV3       InceptionResNetV2    MobileNet       MobileNetV2

        if  keras_version > 2.4
        EfficientNetB0     EfficientNetB1     EfficientNetB2     EfficientNetB3
        EfficientNetB4     EfficientNetB5     EfficientNetB6     EfficientNetB7 """
    parser.add_argument("--architecture_name", help='architecture name')

    args = parser.parse_args()

    epoch = int(args.epoch) if args.epoch else 3
    number_augmentation = int(args.naug) if args.naug else 3
    bsize = int(args.bsize) if args.bsize else 100
    max_sample = int(args.max_sample) if args.max_sample else 1000
    architecture_name = str(args.architecture_name) if args.architecture_name else 'DenseNet121'

    return epoch, bsize, max_sample, architecture_name, number_augmentation


""" Model training and validation """
def architecture(architecture_name='DenseNet121', input_shape=[224, 224, 3], num_classes=14, activation='sigmoid', first_index_trainable=None, weights='imagenet'):

    def custom_model(input_tensor, num_classes):
        model = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), activation='relu')(input_tensor)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.MaxPooling2D(2, 2)(model)

        model = tf.keras.layers.Conv2D(8, kernel_size=(3, 3), activation='relu')(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.MaxPooling2D(2, 2)(model)

        model = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu')(model)
        model = tf.keras.layers.BatchNormalization()(model)
        model = tf.keras.layers.MaxPooling2D(2, 2)(model)

        model = tf.keras.layers.Flatten()(model)
        model = tf.keras.layers.Dense(32, activation='relu')(model)
        model = tf.keras.layers.Dense(num_classes, activation='softmax')(model)

        return tf.keras.models.Model(inputs=model.input, outputs=[model])

    input_tensor = tf.keras.layers.Input(input_shape)

    if architecture_name == 'custom':
        return custom_model(input_tensor, num_classes)


    pooling            = 'avg'
    include_top        = False
    model_architecture = tf.keras.applications.DenseNet121 # The default architecture

    if   architecture_name == 'xception'         : model_architecture = tf.keras.applications.Xception
    elif architecture_name == 'VGG16'            : model_architecture = tf.keras.applications.VGG16
    elif architecture_name == 'VGG19'            : model_architecture = tf.keras.applications.VGG19
    elif architecture_name == 'ResNet50'         : model_architecture = tf.keras.applications.ResNet50
    elif architecture_name == 'ResNet50V2'       : model_architecture = tf.keras.applications.ResNet50V2
    elif architecture_name == 'ResNet101'        : model_architecture = tf.keras.applications.ResNet101
    elif architecture_name == 'ResNet101V2'      : model_architecture = tf.keras.applications.ResNet101V2
    elif architecture_name == 'ResNet152'        : model_architecture = tf.keras.applications.ResNet152
    elif architecture_name == 'ResNet152V2'      : model_architecture = tf.keras.applications.ResNet152V2
    elif architecture_name == 'InceptionV3'      : model_architecture = tf.keras.applications.InceptionV3
    elif architecture_name == 'InceptionResNetV2': model_architecture = tf.keras.applications.InceptionResNetV2
    elif architecture_name == 'MobileNet'        : model_architecture = tf.keras.applications.MobileNet
    elif architecture_name == 'MobileNetV2'      : model_architecture = tf.keras.applications.MobileNetV2
    elif architecture_name == 'DenseNet121'      : model_architecture = tf.keras.applications.DenseNet121
    elif architecture_name == 'DenseNet169'      : model_architecture = tf.keras.applications.DenseNet169
    elif architecture_name == 'DenseNet201'      : model_architecture = tf.keras.applications.DenseNet201

    elif int(list(tf.keras.__version__)[2]) >= 4:

        if   architecture_name == 'EfficientNetB0': model_architecture = tf.keras.applications.EfficientNetB0
        elif architecture_name == 'EfficientNetB1': model_architecture = tf.keras.applications.EfficientNetB1
        elif architecture_name == 'EfficientNetB2': model_architecture = tf.keras.applications.EfficientNetB2
        elif architecture_name == 'EfficientNetB3': model_architecture = tf.keras.applications.EfficientNetB3
        elif architecture_name == 'EfficientNetB4': model_architecture = tf.keras.applications.EfficientNetB4
        elif architecture_name == 'EfficientNetB5': model_architecture = tf.keras.applications.EfficientNetB5
        elif architecture_name == 'EfficientNetB6': model_architecture = tf.keras.applications.EfficientNetB6
        elif architecture_name == 'EfficientNetB7': model_architecture = tf.keras.applications.EfficientNetB7

    model = model_architecture(weights=weights, include_top=include_top, input_tensor=input_tensor, input_shape=input_shape, pooling=pooling)  # ,classes=num_classes

    assert (first_index_trainable < 0) or (first_index_trainable is None), 'first_index_trainable must be negative'

    if first_index_trainable:
        for layer in model.layers[:first_index_trainable]:
            layer.trainable = False

        for layer in model.layers[first_index_trainable:]:
            layer.trainable = True

    KK = tf.keras.layers.Dense(num_classes, activation=activation, name='predictions')(model.output)

    return tf.keras.models.Model(inputs=model.input, outputs=KK)


def weighted_bce_loss(W):
    def func_loss(y_true, y_pred):
        NUM_CLASSES = y_pred.shape[1]

        loss = 0

        for d in range(NUM_CLASSES):
            y_true = tf.cast(y_true, tf.float32)

            # mask   = tf.keras.backend.cast( tf.keras.backend.not_equal(y_true[:,d], -5),
            #                                 tf.keras.backend.floatx() )

            # loss  += W[d]*tf.keras.losses.binary_crossentropy( y_true[:,d] * mask,
            #                                                    y_pred[:,d] * mask )

            loss += W[d] * tf.keras.losses.binary_crossentropy(y_true[:, d], y_pred[:, d])  # type: ignore

        return tf.divide(loss, tf.cast(NUM_CLASSES, tf.float32))

    return func_loss


def optimize(dir_save, data_loader, epochs, architecture_name='DenseNet121', activation='sigmoid', first_index_trainable=None, use_multiprocessing=True, model_metrics=[tf.keras.metrics.binary_accuracy], weights='imagenet', num_classes=0):

    # architecture
    model = architecture(   architecture_name     = architecture_name,
                            input_shape           = list(data_loader.target_size) + [3],
                            num_classes           = num_classes          ,
                            activation            = activation           ,
                            first_index_trainable = first_index_trainable,
                            weights               = weights              )

    model.compile(  optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss      = weighted_bce_loss(data_loader.class_weights),
                    metrics   = model_metrics )

    # optimization
    callbacks = func_callBacks( dir_save = dir_save + 'best_model.h5',
                                mode     = 'min',
                                monitor  = 'val_loss')

    history = model.fit(data_loader.generators['train_with_augments'],
                        validation_data     = data_loader.generators['valid'],
                        epochs              = epochs,
                        steps_per_epoch     = data_loader.steps_per_epoch['train'],
                        validation_steps    = data_loader.steps_per_epoch['valid'],
                        verbose             = 1,
                        use_multiprocessing = use_multiprocessing,
                        callbacks           = callbacks)

    # saving the optimized model
    model.save( dir_save + 'model.h5',
                overwrite         = True,
                include_optimizer = False)

    return model, history


""" Aim1.1: Taxonomy-based Loss Function """


def measure_loss_acc_on_test_data(data, model, labels):
    NUM_CLASSES = len(labels)

    # Chest dataset valid and test were generator while CIFAR360 is a tuple
    data_type_generator = not isinstance(data, (tuple, list))

    if data_type_generator:
        data.reset()

    L = len(data.filenames) if data_type_generator else data[0].shape[0]

    score_values = {}
    for j in tqdm(range(L)):

        # Chest-xray dataset
        if data_type_generator:
            x_test, y_test = next(data)
            full_path, x, y = data.filenames[j], x_test[0, ...], y_test[0, ...]
            x, y = x[np.newaxis, :], y[np.newaxis, :]

        # CIFAR100 dataset
        else:
            full_path, x, y = f'{j}', data[0][j:j + 1, :], data[1][j:j + 1, :]

        # Estimating the loss & accuracy for instance
        eval_results = model.evaluate(x=x, y=y, verbose=0, return_dict=True)

        # predicting the labels for instance
        pred = model.predict(x=x, verbose=0)

        # Measuring the loss for each class
        loss_per_class = [tf.keras.losses.binary_crossentropy(y[..., d], pred[..., d]) for d in range(NUM_CLASSES)]

        # saving all the infos
        score_values[full_path] = {'full_path': full_path, 'loss_avg': eval_results['loss'],
                                   'acc_avg': eval_results['binary_accuracy'],
                                   'pred': pred[0], 'pred_binary': pred[0] > 0.5, 'truth': y[0] > 0.5,
                                   'loss': np.array(loss_per_class), 'label_names': labels}

    # converting the outputs into panda dataframe
    df = pd.DataFrame.from_dict(score_values).T

    # resetting the index to integers
    df.reset_index(inplace=True)

    # # dropping the old index column
    df = df.drop(['index'], axis=1)

    return df


def measure_mean_accruacy_chexpert(truth, prediction, how_to_treat_nans):
    """ prediction & truth: num_samples thresh_technique num_classes """

    pred_classes = prediction > 0.5

    # truth_nan_applied = self._truth_with_nan_applied()
    truth_nan_applied = apply_nan_back_to_truth(truth=truth, how_to_treat_nans=how_to_treat_nans)

    # measuring the binary truth labels (the nan samples will be fixed below)
    truth_binary = truth_nan_applied > 0.5

    truth_pred_compare = (pred_classes == truth_binary).astype(float)

    # replacing the nan samples back to their nan value
    truth_pred_compare[np.where(np.isnan(truth_nan_applied))] = np.nan

    # measuring teh average accuracy over all samples after ignoring the nan samples
    accuracy = np.nanmean(truth_pred_compare, axis=0) * 100

    # this is for safety measure; in case one of the classes overall accuracy was also nan. if removed, then the integer format below will change to very long floats
    accuracy[np.isnan(accuracy)] = 0
    accuracy = (accuracy * 10).astype(int) / 10

    return accuracy


def apply_nan_back_to_truth(truth, how_to_treat_nans):
    # changing teh samples with uncertain truth label to nan
    truth[truth == -10] = np.nan

    # how to treat the nan labels in the original dataset before measuring the average accuracy
    if how_to_treat_nans == 'ignore': truth[truth == -5] = np.nan
    elif how_to_treat_nans == 'pos': truth[truth == -5] = 1
    elif how_to_treat_nans == 'neg': truth[truth == -5] = 0
    return truth


def measure_mean_uncertainty_chexpert(truth, uncertainty, how_to_treat_nans='ignore'):  # type: (np.ndarray, np.ndarray, str) -> np.ndarray
    """ uncertainty & truth:  num_samples thresh_technique num_classes """

    # adding the nan values back to arrays
    truth_nan_applied = apply_nan_back_to_truth(truth, how_to_treat_nans)

    # replacing the nan samples back to their nan value
    uncertainty[np.where(np.isnan(truth_nan_applied))] = np.nan

    # measuring teh average accuracy over all samples after ignoring the nan samples
    uncertainty_mean = np.nanmean(uncertainty, axis=0)

    # this is for safety measure; in case one of the classes overall accuracy was also nan. if removed, then the integer format below will change to very long floats
    uncertainty_mean[np.isnan(uncertainty_mean)] = 0
    uncertainty_mean = (uncertainty_mean * 1000).astype(int) / 1000

    return uncertainty_mean


""" Below is also part of AIM1.1 and should be corrected and merged into the above """


class Measure_Accuracy_Aim1_2:

    def __init__(self, model, generator, predict_accuracy_mode=False, how_to_treat_nans='ignore',
                 uncertainty_type='std'):  # type: (tf.keras.models.Model.dtype, tf.keras.preprocessing.image.ImageDataGenerator, bool, str, str) -> None
        """
        how_to_treat_nans:
            ignore: ignoring the nan samples when measuring the average accuracy
            pos: if integer number, it'll treat as postitive
            neg: if integer number, it'll treat as negative """

        self.uncertainty_final = np.array([])
        self.accuracy_final = np.array([])
        self.probs_std_2d = np.array([])
        self.probs_avg_2d = np.array([])
        self.accuracy_all_augs_3d = np.array([])
        self.probs_all_augs_3d = np.array([])
        self.predict_accuracy_mode = predict_accuracy_mode
        self.how_to_treat_nans = how_to_treat_nans
        self.generator = generator
        self.model = model
        self.uncertainty_type = uncertainty_type
        self.truth = np.array([])

        self._setting_params()

    def _setting_params(self):

        self.full_data_length, self.num_classes = self.generator.labels.shape
        self.batch_size           = self.generator.batch_size
        self.number_batches = int(np.ceil(self.full_data_length / self.batch_size))
        self.truth                    = self.generator.labels.astype(float)

    def loop_over_whole_dataset(self):
        """Looping over all batches """

        probs = np.zeros(self.generator.labels.shape)
        accuracy = None

        # Keras_backend.clear_session()
        self.generator.reset()
        np.random.seed(1)

        for batch_index in tqdm(range(self.number_batches), disable=False):
            # extracting the indexes for batch "batch_index"
            self.generator.batch_index = batch_index
            indexes = next(self.generator.index_generator)

            # print('   extracting data -------')
            self.generator.batch_index = batch_index
            x, _ = next(self.generator)

            # print('   predicting the labels -------')
            probs[indexes, :] = self.model.predict(x, verbose=0)

        # Measuring the accuracy over whole augmented dataset
        if self.predict_accuracy_mode:
            accuracy = measure_mean_accruacy_chexpert(truth=self.truth.copy(), prediction=probs.copy(),
                                                      how_to_treat_nans=self.how_to_treat_nans)

        return probs, accuracy

    def loop_over_all_augmentations(self, number_augmentation: int = 0):

        self.probs_all_augs_3d = np.zeros((1 + number_augmentation, self.full_data_length, self.num_classes))
        self.accuracy_all_augs_3d = np.zeros((1 + number_augmentation, self.num_classes))

        # Looping over all augmentation scenarios
        for ix_aug in range(number_augmentation):
            print(f'augmentation {ix_aug}/{number_augmentation}')
            probs, accuracy = self.loop_over_whole_dataset()

            self.probs_all_augs_3d[ix_aug, ...] = probs
            self.accuracy_all_augs_3d[ix_aug, ...] = accuracy

        # measuring the average probability over all augmented data
        self.probs_avg_2d = np.mean(self.probs_all_augs_3d, axis=0)

        if self.uncertainty_type == 'std':
            self.probs_std_2d = np.std(self.probs_all_augs_3d, axis=0)

        # Measuring the accruacy for new estimated probability for each sample over all augmented data

        # self.accuracy_final    = self._measure_mean_accruacy(self.probs_avg_2d)
        # self.uncertainty_final = self._measure_mean_std(self.probs_std_2d)

        self.accuracy_final = measure_mean_accruacy_chexpert(truth=self.truth.copy(),
                                                             prediction=self.probs_avg_2d.copy(),
                                                             how_to_treat_nans=self.how_to_treat_nans)
        self.uncertainty_final = measure_mean_uncertainty_chexpert(truth=self.truth.copy(),
                                                                   uncertainty=self.probs_std_2d.copy(),
                                                                   how_to_treat_nans=self.how_to_treat_nans)


def apply_technique_aim_1_2(data_generator, data_generator_aug, how_to_treat_nans='ignore', model='', number_augmentation=3, uncertainty_type='std'):
    print('running the evaluation on original non-augmented data')

    MA = Measure_Accuracy_Aim1_2(predict_accuracy_mode=True,
                                 generator=data_generator,
                                 model=model,
                                 how_to_treat_nans=how_to_treat_nans,
                                 uncertainty_type=uncertainty_type)

    probs_2d_orig, old_accuracy = MA.loop_over_whole_dataset()

    print(' running the evaluation on augmented data including the uncertainty measurement')

    MA = Measure_Accuracy_Aim1_2(predict_accuracy_mode=True,
                                 generator=data_generator_aug,
                                 model=model,
                                 how_to_treat_nans=how_to_treat_nans,
                                 uncertainty_type=uncertainty_type)

    MA.loop_over_all_augmentations(number_augmentation=number_augmentation)

    final_results = {'old-accuracy': old_accuracy,
                     'new-accuracy': MA.accuracy_final,
                     'std': MA.uncertainty_final}

    return probs_2d_orig, final_results, MA


def estimate_maximum_and_change(all_accuracies, label_names=None):  # type: (np.ndarray, list) -> pd.DataFrame

    if label_names is None: label_names = []

    columns = ['old-accuracy', 'new-accuracy', 'std']

    # creating a dataframe from accuracies
    df = pd.DataFrame(all_accuracies, index=label_names)

    # adding the 'maximum' & 'change' columns
    df['maximum'] = df.columns[df.values.argmax(axis=1)]
    df['change'] = df[columns[1:]].max(axis=1) - df[columns[0]]

    # replacing "0" values to "--" for readability
    df.maximum[df.change == 0.0] = '--'
    df.change[df.change == 0.0] = '--'

    return df


""" Aim1.3: Soft-weighted MV """


class AIM1_3:
    PROPOSED_METHODS = ['proposed', 'proposed_penalized']

    def __init__(self, data, feature_columns, num_simulations=20, num_labelers=13, low_dis=0.4, high_dis=1):

        self.weights_Tao_mean = None
        self.seed = None
        self.F = None
        self.accuracy = None
        self.weights_Tao = None
        self.prob_weighted = None
        self.weights_proposed = None
        self.labelers_strength = None
        self.true_labels = None
        self.uncertainty_all = None
        self.delta_benchmark = None
        self.delta_proposed = None
        self.predicted_labels_all = None
        self.data = data
        self.num_simulations = num_simulations
        self.feature_columns = feature_columns
        self.num_labelers = num_labelers
        self.low_dis = low_dis
        self.high_dis = high_dis

    def core_measurements(self):  # sourcery skip: raise-specific-error
        """ Final pred labels & uncertainty for proposed technique
                df = predicted_labels[train, test] * [mv]              <=> {rows: samples,  columns: labelers}
                df = uncertainty[train, test]  {rows: samples,  columns: labelers}

            Final pred labels for proposed benchmarks
                df = predicted_labels[train, test] * [simulation_0]    <=> {rows: samples,  columns: labelers} """

        def aim1_3_meauring_probs_uncertainties():
            """ Final pred labels & uncertainty for proposed technique
                    df = predicted_labels[train, test] * [mv]              <=> {rows: samples,  columns: labelers}
                    df = uncertainty[train, test]  {rows: samples,  columns: labelers}

                Final pred labels for proposed benchmarks
                    df = predicted_labels[train, test] * [simulation_0]    <=> {rows: samples,  columns: labelers} """

            def getting_noisy_manual_labels_for_each_worker(true, labelers_strength=0.5, seed_num=1):

                # setting the random seed
                # np.random.seed(seed_num)

                # number of samples and labelers/workers
                num_samples = true.shape[0]

                # finding a random number for each instance
                true_label_assignment_prob = np.random.random(num_samples)

                # samples that will have an inaccurate true label
                false_samples = true_label_assignment_prob < 1 - labelers_strength

                # measuring the new labels for each labeler/worker
                worker_true = true > 0.5
                worker_true[false_samples] = ~ worker_true[false_samples]

                return worker_true

            def assigning_strengths_randomly_to_each_worker():

                labelers_names = [f'labeler_{j}' for j in range(self.num_labelers)]

                labelers_strength_array = np.random.uniform(low=self.low_dis, high=self.high_dis,
                                                            size=self.num_labelers)

                labelers_strength = pd.DataFrame({'labelers_strength': labelers_strength_array}, index=labelers_names)

                return labelers_strength

            def looping_over_all_labelers(labelers_strength):

                """ Looping over all simulations. this is to measure uncertainty """

                predicted_labels_all_sims = {'train': {}, 'test': {}}
                true_labels = {'train': pd.DataFrame(), 'test': pd.DataFrame()}
                uncertainty = {'train': pd.DataFrame(), 'test': pd.DataFrame()}

                for LB_index, LB in enumerate(labelers_strength.index):

                    # Initializationn
                    for mode in ['train', 'test']:
                        predicted_labels_all_sims[mode][LB] = {}
                        true_labels[mode]['truth'] = self.data[mode].true.copy()

                    # Extracting the simulated noisy manual labels based on the worker's strength
                    true_labels['train'][LB] = getting_noisy_manual_labels_for_each_worker(seed_num=0,  # LB_index,
                                                                                           true=self.data['train'].true.values,
                                                                                           labelers_strength=labelers_strength.T[LB].values)

                    true_labels['test'][LB] = getting_noisy_manual_labels_for_each_worker(seed_num=1,  # LB_index,
                                                                                          true=self.data['test'].true.values,
                                                                                          labelers_strength=labelers_strength.T[LB].values)

                    SIMULATION_TYPE = 'random_state'

                    if SIMULATION_TYPE == 'random_state':
                        for sim_num in range(self.num_simulations):
                            # training a random forest on the aformentioned labels
                            RF = sk_ensemble.RandomForestClassifier(n_estimators=4, max_depth=4, random_state=sim_num)  # n_estimators=4, max_depth=4
                            # RF = sklearn.tree.DecisionTreeClassifier(random_state=sim_num)

                            RF.fit(X=self.data['train'][self.feature_columns], y=true_labels['train'][LB])

                            # predicting the labels using trained networks for both train and test data
                            for mode in ['train', 'test']:
                                sim_name = f'simulation_{sim_num}'
                                predicted_labels_all_sims[mode][LB][sim_name] = RF.predict( self.data[mode][self.feature_columns])

                    elif SIMULATION_TYPE == 'multiple_classifier':

                        classifiers_list = [
                            sklearn.neighbors.KNeighborsClassifier(3),  # type: ignore
                            # SVC(kernel="linear", C=0.025),
                            sklearn.svm.SVC(gamma=2, C=1),  # type: ignore
                            # sklearn.gaussian_process.GaussianProcessClassifier(1.0 * sklearn.gaussian_process.kernels.RBF(1.0)),
                            sklearn.tree.DecisionTreeClassifier(max_depth=5),  # type: ignore
                            sk_ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                            sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000),  # type: ignore
                            sk_ensemble.AdaBoostClassifier(),
                            sklearn.naive_bayes.GaussianNB(),  # type: ignore
                            sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(),  # type: ignore
                        ]

                        for sim_num, classifier in enumerate(classifiers_list):

                            classifier.fit(X=self.data['train'][self.feature_columns], y=true_labels['train'][LB])

                            # predicting the labels using trained networks for both train and test data
                            for mode in ['train', 'test']:
                                sim_name = f'simulation_{sim_num}'
                                predicted_labels_all_sims[mode][LB][sim_name] = classifier.predict(
                                    self.data[mode][self.feature_columns])

                    # Measuring the prediction and uncertainty values after MV over all simulations
                    for mode in ['train', 'test']:
                        # converting to dataframe
                        predicted_labels_all_sims[mode][LB] = pd.DataFrame(predicted_labels_all_sims[mode][LB], index=self.data[mode].index)

                        # predicted probability of each class after MV over all simulations
                        predicted_labels_all_sims[mode][LB]['mv'] = ( predicted_labels_all_sims[mode][LB].mean(axis=1) > 0.5)

                        # uncertainty for each labeler over all simulations
                        uncertainty[mode][LB] = predicted_labels_all_sims[mode][LB].std(axis=1)

                # reshaping the dataframes
                predicted_labels = {'train': {}, 'test': {}}
                for mode in ['train', 'test']:

                    # reversing the order of simulations and labelers. NOTE: for the final experiment I should use simulation_0. if I use the mv, then because the augmented truths keeps changing in each simulation, then with enough simulations, I'll end up witht perfect labelers.
                    for i in range(self.num_simulations + 1):

                        SM = f'simulation_{i}' if i < self.num_simulations else 'mv'

                        predicted_labels[mode][SM] = pd.DataFrame()
                        for LB in [f'labeler_{j}' for j in range(self.num_labelers)]:
                            predicted_labels[mode][SM][LB] = predicted_labels_all_sims[mode][LB][SM]

                return true_labels, uncertainty, predicted_labels

            def adding_accuracy_for_each_labeler(labelers_strength, predicted_labels, true_labels):

                labelers_strength['accuracy-test-classifier'] = 0
                labelers_strength['accuracy-test'] = 0

                for i in range(self.num_labelers):
                    LB = f'labeler_{i}'

                    # accuracy of classifier in simulation_0
                    labelers_strength.loc[LB, 'accuracy-test-classifier'] = (
                            predicted_labels['test']['simulation_0'][LB] == true_labels['test'].truth).mean()

                    # accuracy of noisy true labels for each labeler
                    labelers_strength.loc[LB, 'accuracy-test'] = (
                            true_labels['test'][LB] == true_labels['test'].truth).mean()

                return labelers_strength

            # setting a random strength for each labeler/worker
            ls = assigning_strengths_randomly_to_each_worker()

            true_labels, uncertainty, predicted_labels = looping_over_all_labelers(labelers_strength=ls)

            labelers_strength = adding_accuracy_for_each_labeler(labelers_strength=ls,
                                                                 predicted_labels=predicted_labels,
                                                                 true_labels=true_labels)

            return predicted_labels, uncertainty, true_labels, labelers_strength

        def aim1_3_measuring_proposed_weights(predicted_labels, predicted_uncertainty):

            # weights       : num_labelers thresh_technique num_methods
            # prob_weighted : num_samples thresh_technique num_labelers

            # To-Do: This is the part where I should measure the prob_mv_binary for different # of workers instead of all of them
            prob_mv_binary = predicted_labels.mean(axis=1) > 0.5

            T1, T2, w_hat1, w_hat2 = {}, {}, {}, {}

            for workers_name in predicted_labels.columns:
                T1[workers_name] = 1 - predicted_uncertainty[workers_name]

                T2[workers_name] = T1[workers_name].copy()
                T2[workers_name][predicted_labels[workers_name].values != prob_mv_binary.values] = 0

                w_hat1[workers_name] = T1[workers_name].mean(axis=0)
                w_hat2[workers_name] = T2[workers_name].mean(axis=0)

            w_hat = pd.DataFrame([w_hat1, w_hat2], index=AIM1_3.PROPOSED_METHODS).T

            # measuring average weight
            weights = w_hat.divide(w_hat.sum(axis=0), axis=1)

            prob_weighted = pd.DataFrame()
            for method in AIM1_3.PROPOSED_METHODS:
                # prob_weighted[method] =( predicted_uncertainty * weights[method] ).sum(axis=1)
                prob_weighted[method] = (predicted_labels * weights[method]).sum(axis=1)

            return weights, prob_weighted

        def measuring_Tao_weights(delta, noisy_true_labels):
            """
                tau          : 1 thresh_technique 1
                weights_Tao  : num_samples thresh_technique num_labelers
                W_hat_Tao    : num_samples thresh_technique num_labelers
                z            : num_samples thresh_technique 1
                gamma        : num_samples thresh_technique 1
            """

            tau = (delta == noisy_true_labels).mean(axis=0)

            # number of labelers
            M = len(delta.columns)

            # number of true and false labels for each class and sample
            true_counts = delta.sum(axis=1)
            false_counts = M - true_counts

            # measuring the "specific quality of instanses"
            s = delta.multiply(true_counts - 1, axis=0) + (~delta).multiply(false_counts - 1, axis=0)
            gamma = (1 + s ** 2) * tau
            W_hat_Tao = gamma.applymap(lambda x: 1 / (1 + np.exp(-x)))
            z = W_hat_Tao.mean(axis=1)
            weights_Tao = W_hat_Tao.divide(z, axis=0)

            # # Measuring final labels
            # labels            = {}
            # labels['WMV_Tao'] = ( (delta * weights_Tao).mean(axis=1) > 0.5)
            # labels['MV']      = ( delta.mean(axis=1) > 0.5)

            return weights_Tao  # labels

        def measuring_confidence_score(delta_proposed, delta_benchmark, weights_proposed, weights_Tao, true_labels):

            def calculating_confidense_score(delta, weights, conf_strategy, truth):

                def measuring_accuracy(positive_probs, truth):
                    """ Measuring accuracy. This result in the same values as if I had measured a weighted majorith voting using the "weights" multiplied by "delta" which is the binary predicted labels """

                    return ((positive_probs >= 0.5) == truth).mean(axis=0)

                P_pos = (delta * weights).sum(axis=1)
                P_neg = (~delta * weights).sum(axis=1)

                if conf_strategy in (1, 'freq'):

                    F = P_pos.copy()
                    F[P_pos < P_neg] = P_neg[P_pos < P_neg]

                    Accuracy = measuring_accuracy(positive_probs=P_pos, truth=truth)

                elif conf_strategy in (2, 'beta'):

                    f_pos = 1 + P_pos * self.num_labelers
                    f_neg = 1 + P_neg * self.num_labelers

                    k_df = f_pos.floordiv(1)
                    n_df = (f_neg + f_pos).floordiv(1) - 1

                    I = k_df.copy()

                    for index in n_df.index:
                        I[index] = bdtrc(k_df[index], n_df[index], 0.5)

                    # I.hist()

                    F = I.copy()
                    F[I < 0.5] = (1 - I)[I < 0.5]
                    # F.hist()

                    Accuracy = measuring_accuracy(positive_probs=I, truth=truth)
                else:
                    raise Exception('conf_strategy should be either in [1, 2] or ["freq", "beta"]')

                return F, Accuracy

            F, accuracy = {}, {}

            for strategy in ['freq', 'beta']:

                F[strategy], accuracy[strategy] = pd.DataFrame(), pd.DataFrame(index=[self.num_labelers])

                for method in AIM1_3.PROPOSED_METHODS + ['Tao', 'Sheng']:  # Tao: wMV-freq  Sheng: MV-freq

                    if method in AIM1_3.PROPOSED_METHODS:
                        delta = delta_proposed.copy()
                        weights = weights_proposed[method]

                    else:
                        delta = delta_benchmark.copy()
                        if   method in ['Tao'  ]: weights = weights_Tao    / self.num_labelers
                        elif method in ['Sheng']: weights = pd.DataFrame(1 / self.num_labelers, index=weights_Tao.index, columns=weights_Tao.columns)
                        else: raise Exception('method should be either in ["Tao", "Sheng"]')

                    F[strategy][method], accuracy[strategy][method] = calculating_confidense_score(delta=delta, weights=weights, conf_strategy=strategy, truth=true_labels[ 'test'].truth)

                accuracy[strategy]['MV_Classifier'] = ( (delta_benchmark.mean(axis=1) > 0.5) == true_labels['test'].truth).mean(axis=0)

            return F, accuracy

        self.predicted_labels_all, self.uncertainty_all, self.true_labels, self.labelers_strength = aim1_3_meauring_probs_uncertainties()

        self.delta_proposed  = self.predicted_labels_all['test']['mv'          ]
        self.delta_benchmark = self.predicted_labels_all['test']['simulation_0']

        # Measuring weights for the proposed technique
        self.weights_proposed, self.prob_weighted = aim1_3_measuring_proposed_weights( predicted_labels=self.delta_proposed, predicted_uncertainty=self.uncertainty_all['test'])

        # Benchmark accuracy measurement
        self.weights_Tao = measuring_Tao_weights(delta=self.delta_benchmark, noisy_true_labels=self.true_labels['test'].drop(columns=['truth']))

        self.F, self.accuracy = measuring_confidence_score( true_labels     =self.true_labels     ,
                                                            delta_proposed  =self.delta_proposed  ,
                                                            delta_benchmark =self.delta_benchmark ,
                                                            weights_proposed=self.weights_proposed,
                                                            weights_Tao     =self.weights_Tao     )

        return self.true_labels, self.labelers_strength, self.F, self.accuracy, self.weights_proposed, self.weights_Tao

    def applying_other_benchmarks(self):  # , true_labels, F, accuracy

        # Measuring accuracy for other benchmarks
        # crowd_labels = {'train': predicted_labels['train']['simulation_0'],
        # 'test':  predicted_labels['test']['simulation_0']} # aka delta_benchmark

        ground_truth = {'train': self.true_labels['train'].truth.copy(),
                        'test':  self.true_labels['test' ].truth.copy()}

        crowd_labels = {'train': self.true_labels['train'].drop(columns=['truth']).copy(),
                        'test':  self.true_labels['test' ].drop(columns=['truth']).copy()}  # aka delta_benchmark

        ABTC = Aim1_3_ApplyingBenchmarksToCrowdData(crowd_labels=crowd_labels, ground_truth=ground_truth)

        ABTC.apply_all_benchmarks()

        for method in ABTC.benchmarks:
            for strategy in ['freq', 'beta']:
                self.accuracy[strategy][method] = ABTC.accuracy        [method].copy()
                self.F       [strategy][method] = ABTC.aggregatedLabels[method].copy()

    def full_accuracy_comparison(self, seed=0):

        # Setting the random seed
        self.seed = seed
        np.random.seed(seed + 1)

        self.core_measurements()

        self.applying_other_benchmarks()

        # merge labelers_strength and weights
        self.weights_Tao_mean = self.weights_Tao.mean().to_frame().rename(columns={0: 'Tao'})
        self.labelers_strength = pd.concat( [self.labelers_strength, self.weights_proposed * self.num_labelers, self.weights_Tao_mean], axis=1)

        return self.F, self.accuracy, self.labelers_strength


class Aim1_3_ApplyingBenchmarksToCrowdData:
    """
    List of all benchmarks:
        GoldMajorityVote,
        MajorityVote,
        DawidSkene,
        MMSR,
        Wawa,
        ZeroBasedSkill,
        GLAD


    @click.command()
    @click.option('--dataset-name', default='ionosphere', help='Name of the dataset to be used')
    def main(dataset_name = 'ionosphere'):

        # Loading the dataset
        data, feature_columns = load_data.aim1_3_read_download_UCI_database(WhichDataset=dataset_name)



        # generating the noisy true labels for each crowd worker

        ARLS = {'num_labelers':10,  'low_dis':0.3,   'high_dis':0.9}

        predicted_labels, uncertainty, true_labels, labelers_strength = aim1_3_meauring_probs_uncertainties( data = data, ARLS = ARLS, num_simulations = 20,  feature_columns = feature_columns)



        # Finding the accuracy for all benchmark techniques

        ABTC = Aim1_3_ApplyingBenchmarksToCrowdData(true_labels=true_labels['train'] , num_labelers=ARLS['num_labelers'])

        ABTC.apply_all_benchmarks()

        return ABTC.accuracy, ABTC.f1_score

    accuracy, f1_score = main()
    """

    def __init__(self, crowd_labels, ground_truth):

        # self.true_labels  = true_labels
        self.aggregatedLabels = None
        self.f1_score         = None
        self.accuracy         = None
        self.num_labelers     = crowd_labels['test'].columns.shape[0]
        self.benchmarks = ['GoldMajorityVote', 'MajorityVote', 'MMSR', 'Wawa', 'ZeroBasedSkill', 'GLAD', 'DawidSkene']

        self.ground_truth = ground_truth
        self.crowd_labels = crowd_labels

        for mode in ['train', 'test']:
            self.crowd_labels[mode] = self.reshape_dataframe_into_this_sdk_format(self.crowd_labels[mode])

    def apply_all_benchmarks(self):
        """ Apply all benchmarks to the input dataset and return the accuracy and f1 score """

        train    = self.crowd_labels['train']
        train_gt = self.ground_truth['train']
        test     = self.crowd_labels['test' ]

        # Measuring predicted labels for each benchmar technique:
        test_unique = test.task.unique()

        def exception_handler(func):
            def inner_function(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    return np.zeros(test_unique.shape)

            return inner_function

        @exception_handler
        def GoldMajorityVote(train, train_gt, test):
            return crowdkit_aggregation.GoldMajorityVote().fit(train, train_gt).predict(test)

        @exception_handler
        def MajorityVote(train, train_gt, test):
            return crowdkit_aggregation.MajorityVote().fit_predict(test)

        @exception_handler
        def MMSR(train, train_gt, test):
            return crowdkit_aggregation.MMSR(n_iter=5).fit(train).predict(test)

        @exception_handler
        def Wawa(train, train_gt, test):
            return crowdkit_aggregation.Wawa().fit_predict(test)

        @exception_handler
        def ZeroBasedSkill(train, train_gt, test):
            return crowdkit_aggregation.ZeroBasedSkill(n_iter=5).fit_predict(test)

        @exception_handler
        def GLAD(train, train_gt, test):
            return crowdkit_aggregation.GLAD(n_iter=5).fit_predict(test)

        @exception_handler
        def DawidSkene(train, train_gt, test):
            return crowdkit_aggregation.DawidSkene(n_iter=5).fit_predict(test)

        self.aggregatedLabels = {}
        for bench in self.benchmarks:
            self.aggregatedLabels[bench] = eval(bench)(train, train_gt, test)

        # self.aggregatedLabels['GoldMajorityVote'] = GoldMajorityVote(train, train_gt, test)
        # self.aggregatedLabels['MajorityVote']     = MajorityVote(test)
        # self.aggregatedLabels['MMSR']             = MMSR(train, train_gt, test)
        # self.aggregatedLabels['Wawa']             = Wawa(train, train_gt, test)
        # self.aggregatedLabels['ZeroBasedSkill']   = ZeroBasedSkill(train, train_gt, test)
        # self.aggregatedLabels['GLAD']             = GLAD(train, train_gt, test)
        # self.aggregatedLabels['DawidSkene']       = DawidSkene(train, train_gt, test)

        self.aggregatedLabels = pd.DataFrame.from_dict(self.aggregatedLabels)

        # Measuring the Accuracy & F1-score for each benchmark:
        df_empty = pd.DataFrame([self.num_labelers], columns=['num_labelers']).set_index('num_labelers')

        self.accuracy = df_empty.copy()
        self.f1_score = df_empty.copy()

        # iterate through the benchmarks
        for benchmark in self.benchmarks:

            aggregated_label = self.aggregatedLabels[benchmark]

            try:
                self.accuracy[benchmark] = sk_metrics.accuracy_score(self.ground_truth['test'], aggregated_label)
            except Exception as e:
                print('Error in accuracy_score', benchmark, e)
                self.accuracy[benchmark] = 0

            try:
                self.f1_score[benchmark] = sk_metrics.f1_score(self.ground_truth['test'], aggregated_label)
            except Exception as e:
                print('Error in f1_score', benchmark, e)
                self.f1_score[benchmark] = 0

        return self.accuracy, self.f1_score

    @staticmethod
    def reshape_dataframe_into_this_sdk_format(df_predicted_labels):
        """  Preprocessing the data to adapt to the sdk structure:
        """

        # Converting labels from binary to integer
        df_crowd_labels = df_predicted_labels.astype(int).copy()

        # Separating the ground truth labels from the crowd labels
        # ground_truth = df_crowd_labels.pop('truth')

        # Stacking all the labelers labels into one column
        df_crowd_labels = df_crowd_labels.stack().reset_index().rename( columns={'level_0': 'task', 'level_1': 'performer', 0: 'label'})

        # Reordering the columns to make it similar to crowd-kit examples
        df_crowd_labels = df_crowd_labels[['performer', 'task', 'label']]

        return df_crowd_labels  # , ground_truth


class AIM1_3_Plot:
    """ Plotting the results"""

    def __init__(self, plot_data: pd.DataFrame):

        self.weight_strength_relation_interpolated = None
        assert type(plot_data) == pd.DataFrame, 'plot_data must be a pandas DataFrame'

        self.plot_data = plot_data

    # def plot(self, plot_data: pd.DataFrame, xlabel='', ylabel='', xticks=True, title='', legend=None, smooth=True, show_markers=True):
    def plot(self, xlabel='', ylabel='', xticks=True, title='', legend=None, smooth=True, interpolation_pt_count=1000, show_markers='proposed'):

        columns = self.plot_data.columns.to_list()
        y       = self.plot_data.values.astype(float)
        x       = self._fixing_x_axis(index=self.plot_data.index)

        xnew, y_smooth = data_interpolation(x=x, y=y, smooth=smooth, interpolation_pt_count=interpolation_pt_count)

        self.weight_strength_relation_interpolated = pd.DataFrame(y_smooth, columns=columns, index=xnew)
        self.weight_strength_relation_interpolated.index.name = 'labelers_strength'

        plt.plot(xnew, y_smooth)
        self._show_markers(show_markers=show_markers, columns=columns, x=x, y=y)

        self._show(x=x, xnew=xnew, y_smooth=y_smooth, xlabel=xlabel, ylabel=ylabel, xticks=xticks, title=title, )
        self._legend(legend=legend, columns=columns)

    @staticmethod
    def _show(x, xnew, y_smooth, xlabel, ylabel, xticks, title):

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()

        if xticks:
            plt.xticks(xnew)

        plt.show()

        if xticks:
            plt.xticks(x)

        plt.ylim(y_smooth.min() - 0.1, max(1, y_smooth.max()) + 0.1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)

    @staticmethod
    def _legend(legend, columns):

        if legend is None:
            pass
        elif legend == 'empty':
            plt.legend()
        else:
            plt.legend(columns, **legend)

    @staticmethod
    def _fixing_x_axis(index):
        return index.map(lambda x: int(x.replace('NL', ''))) if isinstance(index[0], str) else index.to_numpy()

    @staticmethod
    def _show_markers(show_markers, columns, x, y):
        if show_markers in ('proposed', True):
            cl = [i for i, x in enumerate(columns) if ('proposed' in x) or ('method' in x)]
            plt.plot(x, y[:, cl], 'o')

        elif show_markers == 'all':
            plt.plot(x, y, 'o')


def data_interpolation(x, y, smooth=False, interpolation_pt_count=1000):
    xnew, y_smooth = x, y

    if smooth:
        SMOOTH_METHOD = 'kernel_regression'

        try:

            if SMOOTH_METHOD == 'spline':

                xnew = np.linspace(x.min(), x.max(), interpolation_pt_count)
                spl = make_interp_spline(x, y, k=2)
                y_smooth = spl(xnew)

            elif SMOOTH_METHOD == 'conv':

                filter_size = 5
                filter_array = np.ones(filter_size) / filter_size
                xnew = x.copy()
                y_smooth = np.zeros(list(xnew.shape) + [2])
                for j in range(y.shape[1]):
                    y_smooth[:, j] = np.convolve(y[:, j], filter_array, mode='same')

            # elif SMOOTH_METHOD == 'kernel_regression':

            #     xnew = np.linspace(thresh_technique.min(), thresh_technique.max(), interpolation_pt_count)
            #     y_smooth = np.zeros(list(xnew.shape) + [y.shape[1]])
            #     for j in range(y.shape[1]):
            #         kr = statsmodels.nonparametric.kernel_regression.KernelReg(y[:, j], thresh_technique, 'c')
            #         y_smooth[:, j], _ = kr.fit(xnew)

        except Exception as e:
            print(e)
            xnew, y_smooth = x, y

    return xnew, y_smooth

class Aim1_3_Data_Analysis_Results(AIM1_3_Plot):

    def __init__(self, data: pd.DataFrame=None, feature_columns: typing.List[str]=[''], dataset_name: str='mushroom', mlflow_setup=None,  re_plot: bool=True,  upload_artifact: bool=True , download_artifacts: bool=False):

        self._dataset_dict = None
        self.nlabelers_list = None
        self.high_dis = None
        self.low_dis = None
        self.num_simulations = None
        self.num_seeds = None
        self.outputs = None
        self.accuracy = dict(freq=pd.DataFrame(), beta=pd.DataFrame())
        self.accuracy_stacked = pd.DataFrame()
        self.weight_strength_relation = pd.DataFrame()
        self.comparison_Tao_stacked = pd.DataFrame()
        self.df_comparison_Tao = pd.DataFrame()
        self.dataset = dataset_name
        self.upload_artifact = upload_artifact
        self.re_plot = re_plot
        self.mlflow_setup = mlflow_setup
        self.LOG_ARTIFACTS = (self.mlflow_setup and self.upload_artifact)
        self.data = data
        self.feature_columns = feature_columns
        self.download_artifacts = download_artifacts

        self.index_F = 0
        self.index_accuracy = 1
        self.index_strength = 2

    def full_analysis(self, outputs_mode='LOADING_FROM_LOCAL'):

        # getting the parameters
        LOAD_OLD_PARAMS = (self.mlflow_setup.MLFLOW_MODE in ('LOADING_OLD_SIMULATION', 'RE_RUNNING_OLD_SIMULATION'))
        self.get_parameters(run=self.mlflow_setup.run, load_old_params=LOAD_OLD_PARAMS)

        # getting the output files
        self.get_outputs(mode=outputs_mode)

        # measuring the average accuracy over all seeds
        self.avg_accuracy_over_all_seeds()

        # measuring the worker strength weight relationship for proposed and Tao
        self.worker_weight_strength_relation(smooth=True, seed=1, num_labelers=20, interpolation_pt_count=1000)

    def get_parameters(self, run, load_old_params=True):

        self.num_seeds = eval(run.data.params['num_seeds']) if load_old_params else 6
        self.num_simulations = eval(run.data.params['num_simulations']) if load_old_params else 10
        self.low_dis = eval(run.data.params['low_dis']) if load_old_params else 0.4
        self.high_dis = eval(run.data.params['high_dis']) if load_old_params else 1
        self.nlabelers_list = eval(run.data.params['nlabelers_list']) if load_old_params else range(3, 10)

        if not load_old_params:
            mlflow.log_params(
                                dict(num_seeds      = self.num_seeds      ,
                                    num_simulations = self.num_simulations,
                                    low_dis         = self.low_dis        ,
                                    high_dis        = self.high_dis       ,
                                    nlabelers_list  = self.nlabelers_list )
                            )

        return self.num_seeds, self.num_simulations, self.low_dis, self.high_dis, self.nlabelers_list

    def get_weights(self, worker_index=-1, seed_ix=0):

        assert hasattr(self, 'outputs'), 'self.outputs attribute does not exist. You need to run "get_outputs" first'

        worker_strength_ix = 2

        return (
            self.outputs[f'NL{self.nlabelers_list[worker_index]}'][seed_ix][worker_strength_ix]
            .sort_values(by=['labelers_strength'], ascending=True)
            .round(decimals=2)
        )

    def get_outputs(self, mode='LOADING_FROM_LOCAL'):

        if mode == 'LOADING_FROM_LOCAL':
            self.outputs = pickle.load(open(f'{self.mlflow_setup.dst_path}/outputs_{self.dataset}.pkl', 'rb'))

        elif mode == 'RUNNING_THE_SIMULATION':

            self.outputs = {}
            for NL in tqdm(self.nlabelers_list, desc='looping through different # labelers'):
                aim1_3 = AIM1_3(data=self.data, num_simulations=self.num_simulations,  feature_columns=self.feature_columns, num_labelers=NL, low_dis=self.low_dis,  high_dis=self.high_dis)

                with multiprocessing.Pool(processes=self.num_seeds) as pool:
                    self.outputs[f'NL{NL}'] = pool.map(aim1_3.full_accuracy_comparison, list(range(self.num_seeds)))

            if self.mlflow_setup is None:
                raise Exception("mlflow_setup can't be empty None")
            else:
                self.mlflow_setup.log_artifact(data=self.outputs, path=f'results/tables/outputs_{self.dataset}.pkl', artifact_path='')

        else:
            raise Exception(f"mode {mode} is not supported")

        return self.outputs

    def worker_weight_strength_relation(self, smooth=True, seed=0, num_labelers=13, interpolation_pt_count=1000):

        np.random.seed(seed + 1)

        aim1_3 = AIM1_3(data=self.data, num_simulations=self.num_simulations, feature_columns=self.feature_columns, num_labelers=num_labelers, low_dis=self.low_dis, high_dis=self.high_dis)
        _, labelers_strength, _, _, weights_proposed, weights_Tao = aim1_3.core_measurements()
        df = weights_proposed.applymap(lambda x: x * num_labelers)

        df['Tao'] = pd.DataFrame(weights_Tao.mean(axis=0), columns=['Tao'])

        df = pd.concat([df, labelers_strength], axis=1).set_index('labelers_strength').sort_index()

        self.weight_strength_relation = df.copy()

        # this stacking is solely for the purpose of plotting the figure for paper
        if smooth:
            df_stacked = self._applying_interpolation(interpolation_pt_count=interpolation_pt_count, df=df[['proposed_penalized', 'Tao']])
        else:
            df_stacked = df[['proposed_penalized', 'Tao']].copy()

        df_stacked = df_stacked.stack().to_frame().reset_index().rename(
            columns={'level_1': 'method', 0: 'measured weight', 'labelers_strength': 'worker strength'})

        self.df_comparison_Tao = df.copy()
        self.comparison_Tao_stacked = df_stacked.copy()

        return df, df_stacked

    @property
    def datasets_names(self) -> dict:
        self._dataset_dict =  {1:'kr-vs-kp',    2:'mushroom',   3:'iris',  4:'spambase',    5:'tic-tac-toe',  6:'sick',     7:'waveform',     8:'car',   9:'vote',   10:'ionosphere'}
        return self._dataset_dict

    def dataset_name(self, dataset_ix): # type: (int) -> str
        return self.datasets_names[dataset_ix]

    def run_full_experiment_for_figures(self , mlflow_mode='LOADING_OLD_SIMULATION'):

        def run_for_one_dataset(dataset_name, mlflow_setup):

            # loading the dataset
            np.random.seed(0)
            data, feature_columns = load_data.aim1_3_read_download_UCI_database(WHICH_DATASET=dataset_name, mode='read_arff')

            # getting the run
            mlflow_setup.get_simulation(run_name=dataset_name, mlflow_mode=mlflow_mode, download_artifacts=self.download_artifacts)

            # running analysis
            aim1_3_jn_da = Aim1_3_Data_Analysis_Results(dataset_name=dataset_name, mlflow_setup=mlflow_setup, re_plot=self.re_plot, upload_artifact=self.upload_artifact, data=data, feature_columns=feature_columns)
            aim1_3_jn_da.full_analysis(outputs_mode='LOADING_FROM_LOCAL')

            return aim1_3_jn_da

        def log_artifacts_to_main_simulation():
            do_log_artifact = lambda data , data_type ,  name: self.mlflow_setup.log_artifact(data=data, data_type=data_type, path='results/tables/' + name , upload_artifact=False,  artifact_path='tables')
            do_log_artifact(data=self.accuracy_stacked,                   data_type='csv',  name='accuracy_stacked.csv')
            do_log_artifact(data=self.accuracy,                                 data_type='dict', name='accuracy.pkl')
            do_log_artifact(data=self.comparison_Tao_stacked,       data_type='csv',  name='comparison_Tao_stacked.csv')
            do_log_artifact(data=self.weight_strength_relation,      data_type='csv',  name='weight_strength_relation.csv')

        def concatenate__results_for_all_datasets(df, dataset_name):
            do_concat = lambda df1,df2: pd.concat((df1, df2.assign(dataset_name=dataset_name)))
            self.accuracy_stacked         = do_concat(self.accuracy_stacked         , df.accuracy_stacked         )
            self.comparison_Tao_stacked   = do_concat(self.comparison_Tao_stacked   , df.comparison_Tao_stacked   )
            self.weight_strength_relation = do_concat(self.weight_strength_relation , df.weight_strength_relation )
            self.accuracy ['freq'] = do_concat(self.accuracy ['freq'], df.accuracy ['freq'])
            self.accuracy ['beta'] = do_concat(self.accuracy ['beta'], df.accuracy ['beta'])

        # setting the experiment for mlflow
        mlflow_setup_for_loading_results = AIM1_3_MLFLOW_SETUP(experiment_name='aim1_3_final_results')

        # Looping over all datasets
        for  dataset_name in tqdm(self.datasets_names.values(), desc='looping through datasets'):

            # Running the analysis for one dataset
            aim1_3_jn_da = run_for_one_dataset(dataset_name=dataset_name, mlflow_setup=mlflow_setup_for_loading_results)

            # Concatenating the results
            concatenate__results_for_all_datasets(df=aim1_3_jn_da, dataset_name=dataset_name)

        # logging the results
        log_artifacts_to_main_simulation()


    @staticmethod
    def _applying_interpolation(df, interpolation_pt_count=1000):

        xnew, y_smooth = data_interpolation(x=df.index, y=df.values, smooth=True,
                                            interpolation_pt_count=interpolation_pt_count)
        weight_strength_relation_interpolated = pd.DataFrame(y_smooth, columns=df.columns, index=xnew)
        weight_strength_relation_interpolated.index.name = 'labelers_strength'

        return weight_strength_relation_interpolated

    def avg_accuracy_over_all_seeds(self):

        # Results for each seed
        self._stacking_all_seeds()
        if self.LOG_ARTIFACTS:
            self.mlflow_setup.log_artifact( data=self.accuracy_seeds, path=f'results/accuracy_seeds_{self.dataset}.pkl',
                                            artifact_path='')

        # Average over all seeds
        self.accuracy = {}
        for strategy in ['freq', 'beta']:
            self.accuracy[strategy] = { f'NL{NL}': self.accuracy_seeds[strategy][f'NL{NL}'].mean() for NL in
                                        self.nlabelers_list}
            self.accuracy[strategy] = pd.DataFrame.from_dict(self.accuracy[strategy], orient='index')
            self.accuracy[strategy].index.rename('nlabelers', inplace=True)

        if self.LOG_ARTIFACTS:
            self.mlflow_setup.log_artifact( data=self.accuracy, path=f'results/accuracy_{self.dataset}.pkl',
                                            artifact_path='')

        # Stacking accuracies for all methods and strategies
        self._stacking_accuracy()

    def _stacking_accuracy(self):
        df_freq_stacked = self.accuracy['freq'].stack().reset_index().rename(
            columns={'level_1': 'method', 0: 'accuracy'})
        df_freq_stacked['strategy'] = 'freq'

        df_beta_stacked = self.accuracy['beta'].stack().reset_index().rename(
            columns={'level_1': 'method', 0: 'accuracy'})
        df_beta_stacked['strategy'] = 'beta'

        self.accuracy_stacked = pd.concat([df_freq_stacked, df_beta_stacked], axis=0)

    def _stacking_all_seeds(self):

        def _subfunc(results, strategy):

            df_acc = pd.DataFrame()

            for seedn in range(len(results)):
                df_acc[seedn] = results[seedn][self.index_accuracy][strategy].reset_index(drop=True).T

            return df_acc.T.rename_axis('seed_num')

        def _stacking_accuracy_for_all_seeds(strategy='freq'):

            df = pd.DataFrame()
            for NL in self.nlabelers_list:
                df2 = self.accuracy_seeds[strategy][f'NL{NL}'].reset_index()
                df2['nlabelers'] = NL
                df = pd.concat([df, df2])

            return df

        self.accuracy_seeds, self.accuracy_seeds_stacked = {}, {}

        for STRATEGY in ['freq', 'beta']:
            self.accuracy_seeds[STRATEGY] = {   f'NL{NL}': _subfunc(results=self.outputs[f'NL{NL}'], strategy=STRATEGY) for
                                                NL in self.nlabelers_list}
            self.accuracy_seeds_stacked[STRATEGY] = _stacking_accuracy_for_all_seeds(strategy=STRATEGY)

        return self.accuracy_seeds

    def _renaming_the_methods(self, df):

        # This class inherits from dict and so the keys can be accessed like attributes
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self

        self.names = AttrDict(  {column: AttrDict({'freq': '', 'beta': ''}) for column in
                                ['proposed', 'proposed_penalized', 'Tao', 'Sheng']})

        def subfunc(strategy):

            if isinstance(df[strategy].index[0], str):
                df[strategy].index = df[strategy].index.map(lambda x: int(x.replace('NL', '')))

            proposed = f'uwMV-{strategy} (proposed)'
            proposed_penalized = f'uwMV-{strategy} (proposed_penalized)'
            Tao = f'wMV-{strategy} (Tao)'
            Sheng = f'MV-{strategy} (Sheng)'

            for column in ['proposed', 'proposed_penalized'] + ['Tao', 'Sheng']:
                self.names[column][strategy] = eval(column)

                df[strategy].rename(columns={column: eval(column)}, inplace=True)
                df[strategy].rename_axis('nlabelers', inplace=True)

        subfunc(strategy='freq')
        subfunc(strategy='beta')

        return df


    def paper_final_results_figure_technique(self , comparison ,  strategy='freq' , method='proposed_penalized'):

        sns.set(font_scale=1.8, palette='colorblind', style='darkgrid', context='paper')

        if comparison in (1, 'proposed_vs_proposed_penalized'):
            df          = self.accuracy_stacked[ self.accuracy_stacked.method.isin(['proposed','proposed_penalized']) & (self.accuracy_stacked.strategy == strategy)  ].reset_index(drop=True)
            suptitle = 'Comparison of proposed vs proposed_penalized accuracies'
            x, path   = 'method' ,  'results/final_figures/boxplot_accuracy_strategy.png'

        elif comparison in (2, 'freq_vs_beta' ):
            df       = self.accuracy_stacked[(self.accuracy_stacked.method==method)].reset_index(drop=True)
            suptitle = 'Accuracy of the proposed method'
            x, path  = 'strategy' , 'results/final_figures/boxplot_accuracy_method_freq.png'

        else:
            print(comparison.value)
            raise ValueError('comparison does not exist')

        g = sns.catplot(x=x ,  y='accuracy', col='dataset_name' ,  col_wrap=3, height=4, aspect=1.2, data=df, kind='box', sharey=False, legend_out=True)
        g.set_titles('{col_name}')
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(suptitle)
        g.tight_layout()

        self.mlflow_setup.log_artifact(data=g, upload_artifact=self.upload_artifact, data_type='figure', artifact_path='final_figures', path=path)

    def paper_final_results_figure_weight_quality_relation(self,  font_scale: float=1.8):

        sns.set( palette='colorblind', style='darkgrid', context='paper', font_scale=font_scale)

        df_temp = self.weight_strength_relation.reset_index().melt(id_vars=['labelers_strength','dataset_name'], value_vars=['proposed_penalized', 'Tao'], var_name='method', value_name='weight')

        p = sns.lmplot(x="labelers_strength", y="weight", col='dataset_name', hue='method', col_wrap=3, height=4, aspect=1.2, data=df_temp, order=3, ci=None, legend_out=True)
        p.set_titles('{col_name}')
        p.set_xlabels("worker's quality")
        p.set_ylabels("estimated weight")
        sns.move_legend(p,  "lower right", bbox_to_anchor=(0.8, 0.1), bbox_transform=p.fig.transFigure)
        p.tight_layout()

        # self.mlflow_setup.log_artifact(data=p, upload_artifact=self.upload_artifact, data_type='figure', artifact_path='final_figures', path='results/final_figures/lmplot_weight_strength_relation.png')

    def paper_final_results_figure_weight_quality_relation_detailed(self , dataset_index):

        dataset_name = self.datasets_names[dataset_index]

        comparison_Tao_stacked = self.comparison_Tao_stacked[self.comparison_Tao_stacked.dataset_name==dataset_name].drop(columns=['dataset_name'])
        weight_strength_relation = self.weight_strength_relation[self.weight_strength_relation.dataset_name==dataset_name].drop(columns=['dataset_name'])

        p = sns.jointplot(data=comparison_Tao_stacked, x="worker strength", y="measured weight", hue="method", kind='scatter', joint_kws={"s": 1}, ratio=5, size=5, space=0.1)
        p.ax_joint.plot(weight_strength_relation[ ['proposed_penalized', 'Tao'] ], 'o')
        p.ax_marg_x.set_title(dataset_name)
        p.ax_joint.legend(loc='lower right')

        self.mlflow_setup.log_artifact(data=p, upload_artifact=self.upload_artifact, data_type='figure', artifact_path='final_figures/weight_strength_relation', path=f'results/final_figures/jointplot_weight_strength_relation/{dataset_name}.png')


    def paper_final_results_figure_kde_proposed_penalized_vs_Tao_Sheng(self, strategy='freq'):

        sns.set(font_scale=1.8, palette='colorblind', style='darkgrid', context='paper')

        df_temp_all = self.accuracy[strategy][['dataset_name', 'Tao', 'Sheng', 'proposed_penalized']]

        p = sns.FacetGrid(data=df_temp_all, col='dataset_name',hue='dataset_name', col_wrap=3, height=4, aspect=1.2, sharex=False, sharey=False , legend_out=True)

        for (i,dataset_name), ax in zip(self.datasets_names.items(), p.axes):

            df_temp = df_temp_all[df_temp_all.dataset_name == dataset_name]
            ax.set_title(dataset_name)
            sns.kdeplot(data=df_temp, shade=True, ax=ax, legend=(i==10))
            ax.set_ylabel('')

        for i in [7,8,9]:
            p.axes.flat[i].set_xlabel('Accuracy')

        for i in [0,3,6,9]:
            p.axes.flat[i].set_ylabel('Density')

        p.fig.subplots_adjust(top=1.5)
        p.fig.suptitle(f'KDE Plot for proposed_penalized vs Tao & Sheng   -   strategy: {strategy}')
        # p.legend(loc='lower right')
        p.tight_layout()


        self.mlflow_setup.log_artifact(data=p, upload_artifact=False, data_type='figure', artifact_path='final_figures', path=f'results/final_figures/kde_plot_accuracy_{strategy}.png')


    def paper_final_results_statistical_table(self, comparison, strategy='freq'):
        df_all_temp = self.accuracy_stacked.groupby(['strategy','method', 'dataset_name'])

        results_all = pd.DataFrame()
        for dataset_name in self.datasets_names.values():

            if comparison in (1, 'proposed_vs_proposed_penalized'):
                df1 = df_all_temp.get_group((strategy, 'proposed'          , dataset_name)).accuracy.rename('proposed'          )
                df2 = df_all_temp.get_group((strategy, 'proposed_penalized', dataset_name)).accuracy.rename('proposed_penalized')
                path = f'results/tables/ttest/proposed_vs_proposed_penalized_for_{strategy}.csv'

            elif comparison in (2, 'freq_vs_beta' ):
                df1  = df_all_temp.get_group(('freq' , 'proposed_penalized' , dataset_name)).accuracy.rename('freq')
                df2  = df_all_temp.get_group(('beta' , 'proposed_penalized' , dataset_name)).accuracy.rename('beta')
                path = 'results/tables/ttest/freq_vs_beta_for_proposed_penalized.csv'

            elif comparison in (3, 'proposed_penalized_vs_Tao'):
                df1 = df_all_temp.get_group((strategy, 'Tao'                , dataset_name)).accuracy.rename('Tao'                )
                df2 = df_all_temp.get_group((strategy, 'proposed_penalized' , dataset_name)).accuracy.rename('proposed_penalized' )
                path = f'results/tables/ttest/proposed_penalized_vs_tao_for_{strategy}.csv'

            _, results = rp.ttest(df1, df2)

            results_all[dataset_name] = results.set_index('Independent t-test').rename(columns={'results':dataset_name})

        # Logging artifacts
        self.mlflow_setup.log_artifact(data=results_all, upload_artifact=self.upload_artifact, data_type='csv', artifact_path='tables/ttest', path=path)

        return results_all.round(decimals=3)

    def paper_final_results_figure_heatmap_all_benchmarks_different_workers(self , dataset, strategy = 'freq' ):

        sns.set( palette='colorblind', style='darkgrid', context='paper', font_scale=1.6)

        dataset_name = dataset if isinstance(dataset , str) else self.datasets_names[dataset]
        df_temp_all = self.accuracy[strategy]

        df = df_temp_all[df_temp_all.dataset_name==dataset_name].drop(columns=['dataset_name' , 'MV_Classifier'])

        fig = plt.figure(figsize=(20,7))
        sns.heatmap(df.T, annot=True, fmt='.2f', cmap='Blues', cbar=True, robust=True)
        plt.xlabel('# workers')
        plt.ylabel('method')
        plt.title(f'Accuracy:     strategy: {strategy}    -    dataset: {dataset_name}')

        self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='final_figures', path=f'results/final_figures/heatmap_accuracy_all_benchmarks_for_{dataset_name}.png')

    def paper_final_results_figure_heatmap_all_benchmarks_different_datasets(self , strategy = 'freq' , num_workers=3):

        sns.set( palette='colorblind', style='darkgrid', context='paper', font_scale=1.6)

        df_temp_all = self.accuracy[strategy].drop(columns=['MV_Classifier'])
        df = df_temp_all[df_temp_all.index == f'NL{num_workers}'].set_index('dataset_name')

        fig = plt.figure(figsize=(20,7))
        sns.heatmap(df.T, annot=True, fmt='.2f', cbar=True, cmap='Blues')
        plt.title(f'Accuracy:     NL{num_workers} ({num_workers} workers) \n')
        plt.ylabel('method')

        self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='final_figures', path=f'results/final_figures/heatmap_accuracy_all_benchmarks_for_NL{num_workers}.png')


    def plot_worker_weight_strength_relation(self, xticks=False, smooth=False, interpolation_pt_count=1000,  legend={'loc': 'lower right'}, show_markers='all'):

        assert (self.weight_strength_relation is not None), 'worker_weight_strength_relation() needs to be ran first'

        AIM1_3_Plot.__init__(self, plot_data=self.weight_strength_relation[['proposed_penalized', 'Tao']])
        self.plot(  xlabel='worker strength',
                    ylabel='measured weights',
                    title=f'Estimated-weight vs Worker-strength  -  Dataset: {self.dataset}',
                    smooth=smooth,
                    legend=legend,
                    xticks=xticks,
                    show_markers=show_markers,
                    interpolation_pt_count=interpolation_pt_count)

    def plot_comparing_proposed_methods_1_2(self, re_plot=True, smooth=True, legend={'loc': 'lower right'}):

        assert (self.accuracy is not None), 'avg_accuracy_over_all_seeds() needs to be ran first'

        fig = plt.figure(figsize=(20, 5))
        path = f'figures/Proposed method 1 vs 2 - Dataset {self.dataset}.jpg'

        if re_plot:

            for ix, strategy in enumerate(['freq', 'beta']):
                plt.subplot(1, 2, ix + 1)
                AIM1_3_Plot.__init__(self, plot_data=self.accuracy[strategy][['proposed', 'proposed_penalized']])
                self.plot(  xlabel='# workers',
                            ylabel='accuracy',
                            title=f'strategy: {strategy}    -     Dataset: {self.dataset}',
                            smooth=smooth,
                            legend=legend)

            if self.LOG_ARTIFACTS:
                self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='figures', path=path)

        else:

            plt.imshow(plt.imread(path))
            plt.axis('off')

    def plot_comparing_proposed_methods_freq_beta(self, re_plot=True, smooth=True, legend={'loc': 'lower right'}):

        assert (self.accuracy is not None), 'avg_accuracy_over_all_seeds() needs to be ran first'

        fig = plt.figure(figsize=(20, 5))
        path = f'figures/Proposed method freq vs beta - Dataset {self.dataset}.jpg'

        if re_plot:

            for ix_method, method in [(1, 'proposed'), (2, 'proposed_penalized')]:
                plt.subplot(1, 2, ix_method)
                AIM1_3_Plot.__init__(self, plot_data=pd.DataFrame.from_dict(
                    {'freq': self.accuracy['freq'][method], 'beta': self.accuracy['beta'][method]}))
                self.plot(  xlabel='# workers',
                            ylabel='accuracy',
                            title=f'METHOD {ix_method} ({method})    -     Dataset: {self.dataset}',
                            smooth=smooth,
                            legend=legend)

            if self.LOG_ARTIFACTS:
                self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='figures', path=path)

        else:

            plt.imshow(plt.imread(path))
            plt.axis('off')

    def plot_comparing_proposed_with_Tao_Sheng_MV(self, re_plot=True, smooth=True, legend={'loc': 'lower right'}):

        fig = plt.figure(figsize=(20, 5))
        path = f'figures/Proposed method comparison to Tao and Sheng - Dataset {self.dataset}.jpg'

        if re_plot:

            for ix, strategy in enumerate(['freq', 'beta']):
                plt.subplot(1, 2, ix + 1)
                AIM1_3_Plot.__init__(self, plot_data=self.accuracy[strategy][
                    ['proposed', 'proposed_penalized', 'Tao', 'Sheng', 'MajorityVote']])
                self.plot(  xlabel='# workers',
                            ylabel='accuracy',
                            title=f'Dataset: {self.dataset}',
                            show_markers='proposed',
                            smooth=smooth,
                            legend=legend)

            if self.LOG_ARTIFACTS:
                self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='figures', path=path)

        else:

            plt.imshow(plt.imread(path))
            plt.axis('off')

    def plot_comparing_proposed_with_all_benchmarks(self, re_plot=True, smooth=True):

        for strategy in ['freq', 'beta']:

            fig = plt.figure(figsize=(14, 5))
            path = f'figures/Proposed method comparison to all benchmarks - Dataset {self.dataset} - {strategy.upper()}.jpg'

            if re_plot:

                columns = ['proposed', 'proposed_penalized', 'Tao', 'Sheng', 'MajorityVote', 'MMSR', 'Wawa', 'ZeroBasedSkill', 'GLAD', 'DawidSkene']

                AIM1_3_Plot.__init__(self, plot_data=self.accuracy[strategy][columns])
                self.plot(  xlabel='# workers',
                            ylabel='accuracy',
                            title=f'Dataset: {self.dataset}   -   {strategy.upper()}',
                            show_markers='proposed',
                            smooth=smooth,
                            legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)})

                if self.LOG_ARTIFACTS:
                    self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='figures', path=path)

            else:

                plt.imshow(plt.imread(path))
                plt.axis('off')

    def plot_new_comparing_proposed_methods(self, re_plot=True):

        fig = plt.figure(figsize=(10, 3))
        path = f'figures/Proposed method comparison - Dataset {self.dataset}.png'

        if re_plot:

            df = self.accuracy_stacked

            plt.subplot(121)
            sns.boxplot(x='strategy', y='accuracy', data=df[df.method == 'proposed'])
            plt.title(f'Method: proposed     -    Dataset: {self.dataset}')

            plt.subplot(122)
            sns.boxplot(x='method', y='accuracy', data=df[df.method.isin(['proposed', 'proposed_penalized']) & (df.strategy == 'freq')])
            plt.title(f'Strategy: freq     -    Dataset: {self.dataset}')

            if self.LOG_ARTIFACTS:
                self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='figures', path=path)

        else:
            plt.imshow(plt.imread(path))
            plt.axis('off')

    def plot_new_worker_weight_strength_relation(self, re_plot=True, smooth=True, seed=1, interpolation_pt_count=1000,  num_labelers=20):

        path = f'figures/Estimated-weight vs Worker-strength - Dataset {self.dataset} - via seaborn.png'

        if re_plot:
            _, comparison_Tao_stacked = self.worker_weight_strength_relation(smooth=smooth, seed=seed,
                                                                                num_labelers=num_labelers,
                                                                                interpolation_pt_count=interpolation_pt_count)

            p = sns.jointplot(  data=comparison_Tao_stacked, x="worker strength", y="measured weight", hue="method",
                                ylim=(0, 1.6), xlim=(0, 1), kind='scatter', joint_kws={"s": 1}, ratio=3, size=7, space=0.1)
            p.ax_joint.plot(self.weight_strength_relation[['proposed_penalized', 'Tao']], 'o')
            p.ax_marg_x.set_title(f'Dataset: {self.dataset}')
            p.ax_joint.legend(loc='lower right')

            if self.LOG_ARTIFACTS:
                self.mlflow_setup.log_artifact(data=p, data_type='figure', artifact_path='figures', path=path)

        else:
            plt.imshow(plt.imread(path))
            plt.axis('off')

    def plot_new_comparing_proposed_with_Tao_Sheng_MV(self, re_plot=True):

        fig = plt.figure(figsize=(20, 5))
        path = f'figures/Accuracy distribution comparison to Tao and Sheng - Dataset {self.dataset}.jpg'

        if re_plot:

            for i, strategy in enumerate(['freq', 'beta']):

                plt.subplot(1, 2, i + 1)
                sns.kdeplot(data=self.accuracy[strategy][['proposed_penalized', 'Tao', 'Sheng']], shade=True,
                            legend=True, cbar=True)
                plt.xlabel('accuracy')
                plt.title(strategy.capitalize())

                if self.LOG_ARTIFACTS:
                    self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='figures', path=path)

        else:
            plt.imshow(plt.imread(path))
            plt.axis('off')

    def plot_new_comparing_proposed_with_all_benchmarks(self, re_plot=True):

        for strategy in ['freq', 'beta']:

            fig = plt.figure(figsize=(20, 7))
            path = f'figures/Proposed method vs all benchmarks  average accuracy - Dataset {self.dataset} - {strategy}.jpg'

            if re_plot:

                df = self.accuracy[strategy].drop(columns=['MV_Classifier'])

                sns.heatmap(df.iloc[:6], annot=True, fmt='.2f', cmap='Blues', cbar=True, robust=True)
                plt.title(f'average accuracy - Dataset {self.dataset}  - {strategy.upper()} \n')
                plt.ylabel('# workers')

                if self.LOG_ARTIFACTS:
                    self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='figures', path=path)

            else:
                plt.imshow(plt.imread(path))
                plt.axis('off')

    def plot_new_comparing_proposed_with_all_benchmarks_distribution(self, re_plot=True):

        for strategy in ['freq', 'beta']:

            fig = plt.figure(figsize=(20, 7))
            path = f'figures/Proposed method vs all benchmarks density function - Dataset {self.dataset} - {strategy}.jpg'

            if re_plot:

                BOUNDARY = max(self.nlabelers_list)
                data = self.accuracy_stacked[self.accuracy_stacked.strategy == 'freq'].copy()
                data.nlabelers = data.nlabelers.map(lambda x: int(x[2:]))

                sns.violinplot(x='method', y='accuracy', data=data[data.nlabelers <= BOUNDARY])
                plt.title(f'average accuracy distribution    -     # workers <= {BOUNDARY} - {strategy.upper()}')

                if self.LOG_ARTIFACTS:
                    self.mlflow_setup.log_artifact(data=fig, data_type='figure', artifact_path='figures', path=path)

            else:
                plt.imshow(plt.imread(path))
                plt.axis('off')
