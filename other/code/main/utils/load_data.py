import pandas as pd
import numpy as np
import tensorflow as tf
import os
import wget
from sklearn import preprocessing
from collections import defaultdict
import tensorflow_datasets as tfds
# from sqlalchemy import column

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


class LOAD_CIFAR100():

    def __init__(self, site='local', dir_dataset=None):

        self.meta = None
        self.hierarchy = None
        self.data_modes = None
        self.coarse_fine_label_map_df = None
        self.coarse_fine_label_map = None
        DIR_DICT = {'hpc': '/home/u29/mohammadsmajdi/projects/chest_xray/dataset/cifar-100/',
                            'local': '/Users/personal-macbook/Documents/PhD/dataset/cifar-100/'}

        self.dir = DIR_DICT[site] if dir_dataset is None else dir_dataset
        self.target_size , self.n_channels , self.num_classes = (32,32) , 3, 120
        self.dataframes = defaultdict()
        self.class_weights = np.ones(self.num_classes)

        self.generators, self.tfDatasets, self.steps_per_epoch = defaultdict(), defaultdict(), defaultdict()

    def load(self, data_mode=['train', 'valid', 'test'], label_type='merged' , approach='manual'):

        if approach == 'manual':
            self.load_cifar100_manually( data_modes=data_mode , label_type=label_type )
            self.get_iterators(data_modes=['train'] , be_augmented=True  , Keras=True, TF=False, batch_size=128)
            self.get_iterators(data_modes=['valid'] , be_augmented=False, Keras=True, TF=False, batch_size=128)

        # IMPORTANT: TODO: This won't work yet. needs to be fixed. it's better to ignore it and either use my manual approach or the tfds version inside SAM technique
        # elif approach == 'tfds':
        #     self.load_cifar100_tfdata()


    @staticmethod
    def load_cifar100_tfdata(X, Y, strategy, mode='fine', batch_size=128):

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

        ds = {'train': train_ds, 'test': test_ds}
        return ds

    def load_cifar100_manually(self, data_modes=['train', 'valid', 'test'] , label_type='merged'):

        self.load_cifar100_raw_data_manually()

        self.coarse_fine_label_map, self.coarse_fine_label_map_df = self.get_coarse_fine_label_map()

        self.data_modes = data_modes if isinstance(data_modes, list) else [data_modes]

        # preprocessing the input data ( reshaping to RGB and normalizing )
        self._preprocess()

        # Converting the labels to categorical
        self._get_labels(label_type=label_type)

        # Setting the parent child hierarchy
        self.hierarchy = self.coarse_fine_label_map['names']

    @staticmethod
    def load_cifar100_raw_data_tfds():

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

    def load_cifar100_raw_data_manually(self):

        def _separate_test_valid(df=None):

            df_test  = df.sample(frac=0.7, random_state=42)
            df_valid = df.drop(df_test.index)

            df_test.reset_index(drop=True, inplace=True)
            df_valid.reset_index(drop=True, inplace=True)

            return df_test, df_valid

        load_from_pickle = lambda x: pd.DataFrame.from_dict( pd.read_pickle( self.dir + x) , orient='index').T

        # Loading the meta data
        self.meta = load_from_pickle('meta')

        # Loading the train data
        df_train_valid = load_from_pickle('train')
        self.dataframes['train'], self.dataframes['valid'] = _separate_test_valid(df=df_train_valid)

        # Loading the test and valid data
        self.dataframes['test'] = load_from_pickle('test')

    def get_coarse_fine_label_map(self):

        # Grouping the dataframe by corase labels
        g = self.dataframes['train'].groupby('coarse_labels')

        # Getting the indices for each coarse label
        coarse_ids_list = g.size().index.values

        # Getting the corresponding fine labels for each coarse label
        coarse_fine_label_map = {'ids':{} , 'names':{}}
        for coarse_id in coarse_ids_list:

            # Fine label IDs that corresponds to the coarse label ID
            coarse_fine_label_map['ids'][coarse_id] = g.get_group(coarse_id).fine_labels.unique()

            coarse_name = self.meta.coarse_label_names[coarse_id]

            # List of fine label names that corresponds to the coarse label ID/name
            fine_names_list  = [ self.meta.fine_label_names[i] for i in coarse_fine_label_map['ids'][coarse_id] ]

            coarse_fine_label_map['names'][coarse_name] = fine_names_list

        coarse_fine_label_map_df = {'ids':{} , 'names':{}}
        for name in [ 'ids' , 'names']:
            coarse_fine_label_map_df[name] = pd.DataFrame(coarse_fine_label_map[name]).T.reset_index().rename(columns={'index':'fine_label'}).set_index('fine_label')

        return coarse_fine_label_map, coarse_fine_label_map_df

    def _preprocess(self, data_reshape_to_RGB=True , normalize=True):

        for dm in self.data_modes:

            # Dropping the batch_label column ( it doesn't contain any information )
            self.dataframes[dm].drop(columns=['batch_label'], inplace=True)

            if data_reshape_to_RGB:
                self.dataframes[dm].data = self.dataframes[dm].data.map( lambda x: x.reshape([3,32,32]).transpose([1,2,0]) )

            if normalize:
                self.dataframes[dm].data = self.dataframes[dm].data / 255.0

    def _get_labels(self, label_type='merged'):

        def merge_fine_coarse_labels(df=None):

            to_categorical = lambda x: tf.keras.utils.to_categorical(x, num_classes=self.num_classes, dtype='int')

            # Original fine_labels are between 0 and 99. This makes it between 20 and 119
            df.fine_labels_categorical = (df.fine_labels + 20).map(to_categorical)

            # Original coarse_labels are between 0 and 19
            df.coarse_labels_categorical = df.coarse_labels.map(to_categorical)

            merged_labels_categorical = df.coarse_labels_categorical + df.fine_labels_categorical

            label_names = self.meta.coarse_label_names[:20].to_list() + self.meta.fine_label_names.to_list()

            return merged_labels_categorical, label_names

        for dm in self.data_modes:

            df = self.dataframes[dm]

            labels_dict = {
                'merged':  merge_fine_coarse_labels(df=df),
                'fine':   ( df.fine_labels    , self.meta.fine_label_names.to_list() ),
                'coarse': ( df.coarse_labels  , self.meta.coarse_label_names[:20].to_list() ) }

            self.dataframes[dm]['labels'] , self.label_names = labels_dict[label_type]

    def get_iterators(self, data_modes=['train', 'valid'], be_augmented=False, Keras=True, TF=False, batch_size=128):

        def create_generator(df):

                if not be_augmented:
                    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

                else:
                    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                                height_shift_range=0.2,
                                width_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                rotation_range=40,
                                horizontal_flip=True,
                                vertical_flip=True,
                                fill_mode='nearest',)

                # Loading the data from physical storage
                x =  np.array(df.data.to_list())
                y =  np.array(df.labels.to_list())

                datagen.fit(x)

                data_generator = datagen.flow(x, y, batch_size=batch_size, shuffle=True)

                steps_per_epoch = int( df.data.shape[0] / batch_size )

                return data_generator, steps_per_epoch

        in_shape   = [ None ] + list(self.target_size) + [self.n_channels]
        out_shape = [ None ] + [ self.num_classes ]

        for dm in data_modes:

            name = dm + '_with_augments' * be_augmented
            if Keras:  self.generators[name] ,  self.steps_per_epoch[dm] = create_generator(df=self.dataframes[dm])

            if Keras and TF:  self.tfDatasets[name]  = tf.data.Dataset.from_generator(lambda: self.generators[name] , output_types=(tf.float32 , tf.float32) , output_shapes=( in_shape , out_shape )  )


class LOAD_CHEST_XRAY():

    def __init__(self, site='hpc', dataset_dir=None, dataset_name='chexpert', batch_size=30, max_sample=100000):

        self.class_weights = None
        self.label_names = None
        self.dataframes = None
        DIR_DICT = {'hpc':   '/groups/jjrodrig/projects/chest/dataset/',
                             'local': '/Users/personal-macbook/Documents/PhD/dataset/'}

        self.dir = DIR_DICT[site] + dataset_name + '/' if dataset_dir is None else dataset_dir

        self.target_size, self.n_channels = ((224,224), 3)

        self.dataset_name = dataset_name
        self.batch_size      = batch_size
        self.max_sample   = max_sample

        self.tfDatasets, self.generators , self.steps_per_epoch = defaultdict() ,defaultdict() ,defaultdict(),

        self.hierarchy = { 'Lung Opacity': ['Pneumonia', 'Atelectasis','Consolidation','Lung Lesion', 'Edema'] ,
                           'Enlarged Cardiomediastinum':  ['Cardiomegaly']  }

        self.pathologies = ["No Finding", "Enlarged Cardiomediastinum" , "Cardiomegaly" , "Lung Opacity" , "Lung Lesion", "Edema" , "Consolidation" , "Pneumonia" , "Atelectasis" , "Pneumothorax" , "Pleural Effusion" , "Pleural Other" , "Fracture" , "Support Devices"]

    def load(self, data_mode='train', approach='manual'):

        self.dataframes, self.label_names, self.class_weights = self.load_preprocessed_dataframes(dataset_name=self.dataset_name)

        if approach == 'manual':
            self._load_manually(data_mode=data_mode)

        elif approach == 'tfds':
            self._load_using_tfrecords(data_mode=data_mode)

    def _load_using_tfrecords(self, data_mode='train'):
        # TODO: This is not completed. It loads all 3 labels (including uncertainty) and not just the binary labels

        def augment(image,label):
            image = tf.image.resize_with_crop_or_pad(image, self.target_size[0]+40 , self.target_size[1]+40 ) # Add 40 pixels of padding
            image = tf.image.random_crop(image, size=list(self.target_size + (self.n_channels,)) ) # Random crop back to 224x224
            image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
            image = tf.clip_by_value(image, 0., 1.)
            return image, label

        def formatting(image, label):
            image = tf.image.convert_image_dtype(image, tf.float32)
            label = tf.cast(label, tf.int32)
            return image, label

        AUTO = tf.data.AUTOTUNE
        # this will prepare the tfrecord files using the downloaded dataset
        config = tfds.download.DownloadConfig(extract_dir=self.dir, manual_dir=self.dir, download_mode=tfds.GenerateMode.REUSE_DATASET_IF_EXISTS)
        chexpert_builder = tfds.builder(name=self.dataset_name)
        chexpert_builder.download_and_prepare(download_config=config)

        # After this, you can load the dataset using the following code
        ds = chexpert_builder.as_dataset(split=data_mode, shuffle_files=True, as_supervised=True, batch_size=self.batch_size)

        # After the tfrecords are built once, we can directly load the dataset using tfds.builder('chexpert').as_dataset().

        ds = (  ds.map(formatting, num_parallel_calls=AUTO) )

        if data_mode == 'train':
            ds = (ds.map(augment, num_parallel_calls=AUTO))

        self.tfDatasets = ( ds.prefetch(AUTO) )
        self.info = chexpert_builder.info

    def _load_manually(self, data_mode='train'):

        if data_mode == 'train':
            self.get_iterators(data_modes=['train'] , be_augmented=True  , Keras=True, TF=False, batch_size=128)
            self.get_iterators(data_modes=['valid'] , be_augmented=False, Keras=True, TF=False, batch_size=128)

        elif data_mode in ('valid' , 'test'):
            self.get_iterators(data_modes=[data_mode] , be_augmented=False  , Keras=True, TF=False, batch_size=180)

    def load_preprocessed_dataframes(self, dataset_name='chexpert'):

        dataframes = defaultdict()

        if dataset_name == 'nih':
            dataframes['train'] , dataframes['valid'] , dataframes['test'] , label_names, class_weights = self.nih(
                path=self.dir, max_sample=self.max_sample)

        elif dataset_name == 'chexpert':
            (dataframes['train'] , dataframes['uncertain'] ) , dataframes['valid']  , dataframes['test']  , label_names , class_weights = self.chexpert(
                dir_dataset=self.dir, max_sample=self.max_sample)

        return dataframes , label_names, class_weights

    @staticmethod
    def nih(path, max_sample):

        """ reading the csv tables """
        all_data       = pd.read_csv(path + '/files/Data_Entry_2017_v2020.csv')
        test_list      = pd.read_csv(path + '/files/test_list.txt', names=['Image Index'])



        """ Writing the relative path """
        all_data['Path']      = 'data/' + all_data['Image Index']
        all_data['full_path'] = path + '/data/' + all_data['Image Index']



        """ Finding the list of all studied pathologies """
        all_data['Finding Labels'] = all_data['Finding Labels'].map(lambda x: x.split('|'))
        # pathologies = set(list(chain(*all_data['Finding Labels'])))



        """ overwriting the order of pathologeis """
        pathologies = ['No Finding', 'Pneumonia', 'Mass', 'Pneumothorax', 'Pleural_Thickening', 'Edema', 'Cardiomegaly', 'Emphysema', 'Effusion', 'Consolidation', 'Nodule', 'Infiltration', 'Atelectasis', 'Fibrosis']



        """ Creating the pathology based columns """
        for name in pathologies:
            all_data[name] = all_data['Finding Labels'].map(lambda x: 1 if name in x else 0)



        """ Creating the disease vectors """
        all_data['disease_vector'] = all_data[pathologies].values.tolist()
        all_data['disease_vector'] = all_data['disease_vector'].map(lambda x: np.array(x))



        """ Selecting a few cases """
        all_data = all_data.iloc[:max_sample,:]



        """ Removing unnecessary columns """
        # all_data = all_data.drop(columns=['OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[thresh_technique',	'y]', 'Follow-up #'])



        """ Delecting the pathologies with at least a minimum number of samples """
        # MIN_CASES = 1000
        # pathologies = [name for name in pathologies if all_data[name].sum()>MIN_CASES]
        # print('Number of samples per class ({})'.format(len(pathologies)),
        #     [(name,int(all_data[name].sum())) for name in pathologies])



        """ Resampling the dataset to make class occurrences more reasonable """
        # CASE_NUMBERS = 800
        # sample_weights = all_data['Finding Labels'].map(lambda thresh_technique: len(thresh_technique) if len(thresh_technique)>0 else 0).values + 4e-2
        # sample_weights /= sample_weights.sum()
        # all_data = all_data.sample(CASE_NUMBERS, weights=sample_weights)



        """ Separating train validation test """
        test      = all_data[all_data['Image Index'].isin(test_list['Image Index'])]
        train_val = all_data.drop(test.index)

        valid     = train_val.sample(frac=0.2,random_state=1)
        train     = train_val.drop(valid.index)

        print('after sample-pruning')
        print('train size:',train.shape)
        print('valid size:',valid.shape)
        print('test size:' ,test.shape)



        """ Class weights """
        L = len(pathologies)
        class_weights = np.ones(L)/L


        return train, valid, test, pathologies, class_weights

    @staticmethod
    def chexpert(dir_dataset, max_sample):

        def cleaning_up_dataframe(data, pathologies_in, mode):
            """ Label Structure
                positive (exist):            1.0
                negative (doesn't exist):   -1.0
                Ucertain                     0.0
                no mention                   nan """

            # changing all no mention labels to negative
            data = data[data['AP/PA']=='AP']
            data = data[data['Frontal/Lateral']=='Frontal']


            # Treat all other nan s as negative
            # data = data.replace(np.nan,-1.0)


            # renaming the pathologeis to 'neg' 'pos' 'uncertain'
            for column in pathologies_in:

                data[column] = data[column].replace(1,'pos')

                if mode == 'train':
                    data[column] = data[column].replace(-1,'neg')
                    data[column] = data[column].replace(0,'uncertain')
                elif mode == 'test':
                    data[column] = data[column].replace(0,'neg')


            # according to CheXpert paper, we can assume all pathologise are negative when no finding label is True
            no_finding_indexes = data[data['No Finding']=='pos'].index
            for disease in pathologies_in:
                if disease != 'No Finding':
                    data.loc[no_finding_indexes, disease] = 'neg'


            return data

        def replacing_parent_nan_values_with_one_if_child_exist(data: pd.DataFrame):

            """     parent ->
                        - child

                    Lung Opacity ->

                        - Pneuomnia
                        - Atelectasis
                        - Edema
                        - Consolidation
                        - Lung Lesion

                    Enlarged Cardiomediastinum ->

                        - Cardiomegaly       """


            func = lambda x1, x2: 1.0 if np.isnan(x1) and x2==1.0 else x1

            for child_name in ['Pneumonia','Atelectasis','Edema','Consolidation','Lung Lesion']:

                data['Lung Opacity'] = data['Lung Opacity'].combine(data[child_name], func=func)


            for child_name in ['Cardiomegaly']:

                data['Enlarged Cardiomediastinum'] = data['Enlarged Cardiomediastinum'].combine(data[child_name], func=func)

            return data



        """ Selecting the pathologies_in """
        pathologies = ["No Finding", "Enlarged Cardiomediastinum" , "Cardiomegaly" , "Lung Opacity" , "Lung Lesion", "Edema" , "Consolidation" , "Pneumonia" , "Atelectasis" , "Pneumothorax" , "Pleural Effusion" , "Pleural Other" , "Fracture" , "Support Devices"]


        """ Loading the raw table """
        # train = pd.read_csv(dataset_dir + '/train_aim1_2.csv')
        train = pd.read_csv(dir_dataset + '/CheXpert-v1.0-small/train.csv')
        test  = pd.read_csv(dir_dataset + '/CheXpert-v1.0-small/valid.csv')

        print('before sample-pruning')
        print('train:',train.shape)
        print('test:',test.shape)

        """ Label Structure
            positive (exist):            1.0
            negative (doesn't exist):   -1.0
            Ucertain                     0.0
            no mention                   nan """

        """ Adding full directory """
        train['full_path'] = dir_dataset + '/' + train['Path']
        test['full_path'] = dir_dataset + '/' + test['Path']



        """ Extracting the pathologies_in of interest """
        train = cleaning_up_dataframe(train, pathologies, 'train')
        test  = cleaning_up_dataframe(test, pathologies , 'test')


        """ Selecting a few cases """
        train = train.iloc[:max_sample,:]
        test  = test.iloc[:max_sample ,:]


        """ Separating the uncertain samples """
        train_uncertain = train.copy()
        for name in pathologies:
            train = train.loc[train[name]!='uncertain']

        train_uncertain = train_uncertain.drop(train.index)


        """ Splitting train/validatiion """
        valid = train.sample(frac=0.2,random_state=1)
        train = train.drop(valid.index)


        print('\nafter sample-pruning')
        print('train (certain):',train.shape)
        print('train (uncertain):',train_uncertain.shape)
        print('valid:',valid.shape)
        print('test:',test.shape,'\n')


        # TODO make no finding 0 for all samples where we at least have one case
        """ Changing classes from string to integer
            Tagging the missing labels; this number "-0.5" will later be masked during measuring the loss """

        train_uncertain = train_uncertain.replace('pos',1).replace('neg',0).replace(np.nan,-5.0).replace('uncertain',-10.0)
        train = train.replace('pos',1).replace('neg',0).replace(np.nan,-5.0)
        valid = valid.replace('pos',1).replace('neg',0).replace(np.nan,-5.0)
        test  = test.replace('pos',1).replace('neg',0)


        """ Changing the nan values for parents with at lease 1 TRUE child to TRUE """
        train_uncertain = replacing_parent_nan_values_with_one_if_child_exist(train_uncertain)
        train = replacing_parent_nan_values_with_one_if_child_exist(train)
        valid = replacing_parent_nan_values_with_one_if_child_exist(valid)



        """ Class weights """
        L = len(pathologies)
        class_weights = np.ones(L)/L

        return (train, train_uncertain), valid, test, pathologies, class_weights

    def get_iterators(self, data_modes=['train'  ,  'valid'  ,  'test' , 'uncertain'], be_augmented=False, Keras=True, TF=False, batch_size=128):

        def create_generator(df):

            if not be_augmented:
                datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

            else:
                datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                            fill_mode='nearest',
                            rescale=1./255,
                            rotation_range=15,
                            height_shift_range=0.1,
                            width_shift_range=0.1,
                            horizontal_flip=False,
                            vertical_flip=False,
                            featurewise_center=False,
                            samplewise_center=False,
                            featurewise_std_normalization=False,
                            samplewise_std_normalization=False,  )

            # Loading the data from physical storage
            data_generator = datagen.flow_from_dataframe(
                        dataframe=df,
                        x_col='Path',
                        y_col=self.label_names,
                        color_mode='rgb',
                        directory=self.dir,
                        target_size=self.target_size,
                        batch_size=batch_size,
                        class_mode='raw',
                        shuffle=False,
                        classes=self.label_names)

            # steps_per_epoch = int(len(data_generator.filenames)/batch_size)
            steps_per_epoch = int( df.shape[0] / batch_size )

            return data_generator, steps_per_epoch

        if not isinstance(data_modes, list):
            data_modes = [data_modes]

        in_shape   = [ None ] + list(self.target_size) + [self.n_channels]
        out_shape = [ None ] + [ len(self.label_names) ]

        for dm in data_modes:

            name = dm + '_with_augments' * be_augmented
            if Keras:  self.generators[name] ,  self.steps_per_epoch[dm] = create_generator(df=self.dataframes[dm])

            if Keras and TF:  self.tfDatasets[name]  = tf.data.Dataset.from_generator(lambda: self.generators[name] , output_types=(tf.float32 , tf.float32) , output_shapes=( in_shape , out_shape )  )


def aim1_3_read_download_UCI_database(WHICH_DATASET='ionosphere', mode='read', dir_all_datasets='datasets/'):

    dir_all_datasets = os.path.abspath(dir_all_datasets)

    if not os.path.isdir(dir_all_datasets):
        raise ValueError('The directory does not exist')

    def read_raw_names_files(WHICH_DATASET='ionosphere'):

        main_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'


        if WHICH_DATASET in (1,'kr-vs-kp'):
            dataset = 'kr-vs-kp'
            names   = [f'a{i}' for i in range(36)] + ['true']
            files   = ['Index', f'{dataset}.data', f'{dataset}.names']
            url     = main_url + '/chess/king-rook-vs-king-pawn/'

        elif WHICH_DATASET in (2,'mushroom'):
            dataset = 'mushroom'
            names   = ['true'] + [f'a{i}' for i in range(22)]
            files   = ['Index', f'{dataset}.data', f'{dataset}.names']
            url     = main_url + '/mushroom/'

        elif WHICH_DATASET in (3,'sick'):
            dataset = 'sick'

            names   = [f'a{i}' for i in range(29)] + ['true']
            files   = [f'{dataset}.data', f'{dataset}.names', f'{dataset}.test']
            url     = main_url + '/thyroid-disease/'

        elif WHICH_DATASET in (4,'spambase'):
            dataset = 'spambase'
            names   = [f'a{i}' for i in range(57)] + ['true']
            files   = [f'{dataset}.DOCUMENTATION', f'{dataset}.data', f'{dataset}.names', f'{dataset}.zip']
            url     = main_url + '/spambase/'

        elif WHICH_DATASET in (5,'tic-tac-toe'):
            dataset = 'tic-tac-toe'
            names   = [f'a{i}' for i in range(9)] + ['true']
            files   = [f'{dataset}.data', f'{dataset}.names']
            url     = main_url + '/tic-tac-toe/'

        elif WHICH_DATASET in (7,'thyroid'):
            pass

        elif WHICH_DATASET in (8,'waveform'):
            dataset = 'waveform'
            names   = [f'a{i}' for i in range(21)] + ['true']
            files   = [ 'Index', f'{dataset}-+noise.c', f'{dataset}-+noise.data.Z', f'{dataset}-+noise.names', f'{dataset}.c', f'{dataset}.data.Z', f'{dataset}.names']
            url     = main_url + '/mwaveform/'

        elif WHICH_DATASET in (9,'biodeg'):
            dataset = 'biodeg'
            names   = [f'a{i}' for i in range(41)] + ['true']
            files   = [f'{dataset}.csv']
            url     = main_url + '/00254/'

        elif WHICH_DATASET in (10,'horse-colic'):
            dataset = 'horse-colic'
            names   = [f'a{i}' for i in range(41)] + ['true']
            files   = [f'{dataset}.data', f'{dataset}.names', f'{dataset}.names.original', f'{dataset}.test']
            url     = main_url + '/horse-colic/'

        elif WHICH_DATASET in (11,'ionosphere'):
            dataset = 'ionosphere'
            names   = [f'a{i}' for i in range(34)] + ['true']
            files   = [ 'Index', f'{dataset}.data', f'{dataset}.names']
            url     = main_url + '/ionosphere/'

        elif WHICH_DATASET in (12,'vote'):
            pass

        return dataset, names, files, url

    def download_data(dir_all_datasets=''):

        dataset, _, files, url = read_raw_names_files(WHICH_DATASET=WHICH_DATASET)

        local_path = f'{dir_all_datasets}/UCI_{dataset}'

        if not os.path.isdir(local_path):
            os.mkdir(local_path)

        for name in files:
            wget.download(url + name, local_path)

        data_raw = pd.read_csv( dir_all_datasets + f'/UCI_{dataset}/{dataset}.data')


        return data_raw, []

    def separate_train_test(data_raw, train_frac=0.8):
        return {'train': data_raw.sample(frac=train_frac).sort_index(),
                'test': data_raw.drop(data['train'].index)}

    def reading_from_arff(dataset):

        def read_data_after_at_data_line(dataset):
            dir_main = os.path.abspath(r'datasets')
            dir_dataset = dir_main + f'/{dataset}/{dataset}.arff'

            with open(dir_dataset, 'r') as f:
                for line in f:
                    if line.lower().startswith('@data'):
                        break

                table = pd.read_csv(f, header=None, sep=',', na_values=['?','nan','null','NaN','NULL'])

            return table

        def changing_str_to_int(data):

            le = preprocessing.LabelEncoder()

            for name in data.columns:
                if data[name].dtype == 'object':
                    data[name] = le.fit_transform(data[name])

            return data

        data = read_data_after_at_data_line(dataset=dataset)

        data = changing_str_to_int(data=data)

        feature_columns = [f'a{i}' for i in range(data.shape[1]-1)]
        data.columns = feature_columns + ['true']

        data.replace( 2147483648, np.nan, inplace=True)
        data.replace(-2147483648, np.nan, inplace=True)

        # removing columns that only has one value (mostly the one that are fully NaN)
        for name in data.columns:
            if len(data[name].unique()) == 1:
                data.drop(columns=name, inplace=True)
                feature_columns.remove(name)

        # extracting only classes "1" and "2" to correspond to Tao et al. paper
        if dataset == 'waveform':
            data = data[data.true != 0]
            data.true.replace({1:0,2:1}, inplace=True)

        if dataset in ('sick', 'hepatitis'):
            data.replace({np.nan:0}, inplace=True)

        if dataset == 'balance-scale':
            data = data[data.true != 1]
            data.true.replace({2:1}, inplace=True)

        if dataset == 'iris':
            data = data[data.true != 2]

        if dataset == 'car':
            data.true.replace({2:1,3:1}, inplace=True) # classes are [unacceptable, acceptable, good, very good]


        return data, feature_columns

    def read_data(dir_all_datasets='', WHICH_DATASET=0):

        def postprocess(data_raw=[], names=[], WHICH_DATASET=0):

            def replacing_classes_char_to_int(data_raw=[], feature_columns=[]):

                # finding the unique classes
                lbls = set()
                for fx in feature_columns:
                    lbls = lbls.union(data_raw[fx].unique())

                # replacing the classes from char to int
                for ix, lb in enumerate(lbls):
                    data_raw[feature_columns] = data_raw[feature_columns].replace(lb,ix+1)

                return data_raw

            feature_columns = names.copy()
            feature_columns.remove('true')

            if WHICH_DATASET in (1,'kr-vs-kp'):

                # changing the true labels from string to [0,1]
                data_raw.true = data_raw.true.replace('won',1).replace('nowin',0)

                # replacing the classes from char to int
                data_raw = replacing_classes_char_to_int(data_raw, feature_columns)

            elif WHICH_DATASET in (2,'mushroom'):

                # changing the true labels from string to [0,1]
                data_raw.true = data_raw.true.replace('e',1).replace('p',0)

                # feature a10 has missing data
                data_raw.drop(columns=['a10'], inplace=True)
                feature_columns.remove('a10')

                # replacing the classes from char to int
                data_raw = replacing_classes_char_to_int(data_raw, feature_columns)

            elif WHICH_DATASET in (3,'sick'):
                data_raw.true = data_raw.true.map(lambda x: x.split('.')[0]).replace('sick',1).replace('negative',0)
                column_name = 'a27' # 'TBG measured'
                data_raw = data_raw.drop(columns=[column_name])
                feature_columns.remove(column_name)

                # replacing the classes from char to int
                # data_raw = replacing_classes_char_to_int(data_raw, feature_columns)

            elif WHICH_DATASET in (4,'spambase'):
                pass

            elif WHICH_DATASET in (5,'tic-tac-toe'):
                # renaming the two classes "good" and "bad" to "0" and "1"
                data_raw.true = data_raw.true.replace('negative',0).replace('positive',1)
                data_raw[feature_columns] = data_raw[feature_columns].replace('thresh_technique',1).replace('o',2).replace('b',0)

            elif WHICH_DATASET in (6, 'splice'):
                pass

            elif WHICH_DATASET in (7,'thyroid'):
                pass

            elif WHICH_DATASET in (8,'waveform'):
                # extracting only classes "1" and "2" to correspond to Tao et al. paper
                class_0 = data_raw[data_raw.true == 0].index
                data_raw.drop(class_0, inplace=True)
                data_raw.true = data_raw.true.replace(1,0).replace(2,1)

            elif WHICH_DATASET in (9,'biodeg'):
                data_raw.true = data_raw.true.replace('RB',1).replace('NRB',0)

            elif WHICH_DATASET in (10,'horse-colic'):
                pass

            elif WHICH_DATASET in (11,'ionosphere'):
                data_raw.true = data_raw.true.replace('g',1).replace('b',0)

            elif WHICH_DATASET in (12,'vote'):
                pass

            return data_raw, feature_columns


        dataset, names, _, _ = read_raw_names_files(WHICH_DATASET=WHICH_DATASET)

        if dataset == 'biodeg':
            command = {'filepath_or_buffer': dir_all_datasets + f'/UCI_{dataset}/{dataset}.csv', 'delimiter':';'}

        elif dataset == 'horse-colic':
            command = {'filepath_or_buffer': dir_all_datasets + f'/UCI_{dataset}/{dataset}.data', 'delimiter':' ', 'index_col':None}

        else:
            command = {'filepath_or_buffer': dir_all_datasets + f'/UCI_{dataset}/{dataset}.data'}

        if mode == 'read':
            data_raw = pd.read_csv(**command, names=names)
            data_raw, feature_columns = postprocess(data_raw=data_raw, names=names, WHICH_DATASET=WHICH_DATASET)

        elif mode == 'read_raw':
            data_raw, feature_columns = pd.read_csv(**command) , []


        data = separate_train_test(data_raw=data_raw, train_frac=0.8)

        return data, feature_columns


    if mode == 'read_arff':

        data, feature_columns = reading_from_arff(WHICH_DATASET)
        data = separate_train_test(data_raw=data, train_frac=0.8)

        return data, feature_columns

    elif  mode == 'download' :
        return download_data(dir_all_datasets=dir_all_datasets)

    elif 'read' in mode:
        return read_data(    dir_all_datasets=dir_all_datasets, WHICH_DATASET=WHICH_DATASET)
