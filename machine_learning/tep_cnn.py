#!/usr/bin/python3

#TODO break up this script into components
# e.g. loading modules that train the model, ....

"""
tep_cnn.py

Train, evaluate, and predict with CNN/PCNN models using Keras/TensorFlow.

This script provides a command-line interface for training, evaluating,
and predicting neural network models on preprocessed data. It supports
flexible configuration via arguments, including model architecture,
data slicing, logging, and checkpointing.

Features:
    - Train CNN/PCNN models on user datasets and features.
    - Evaluate trained models on test data.
    - Predict outputs using trained models.
    - Batch processing and xarray dataset support.
    - Logging and TensorBoard integration.
    - Command-line argument parsing.

Usage:
    python tep_cnn.py --data_dir <path> --data_filenames <files>
        --model_dir <path> --features <list> --target <target>
        --kernels <sizes> --filters <sizes> --epochs <int>
        --batch_size <int> [--train] [--evaluate] [--predict]
        [--pickup] [--verbose]

Arguments: #TODO what is optional and required?
    --data_dir: Directory with preprocessed data files.
    --data_filenames: Space-separated list of data filenames.
    --model_dir: Directory for trained models.
    --model_filename: Model file to load.
    --model_save_filename: Model file to save.
    --features: Space-separated feature names.
    --target: Target variable name.
    --epochs: Number of training epochs.
    --kernels: Space-separated kernel sizes, e.g. "(3,3) (5,5)".
    --filters: Space-separated filter counts.
    --batch_size: Batch size for training/evaluation.
    --train: Train the model.
    --evaluate: Evaluate the model.
    --predict: Predict using the model.
    --pickup: Continue training from checkpoint.
    --verbose: Enable verbose logging.
    --tensboard_log_dir: TensorBoard log directory.
    --verbose_output_filename: Verbose log output file.

History:
    2025-06-23: Initial version.
    2025-07-24: Renamed from train_cnn.py to tep_cnn.py.
    2026-01-22: JASMIN implementation.

Author:
    twilder

Dependencies:
    numpy, xarray, tensorflow, keras, xbatcher, argparse, logging,
    cnn, preprocess_data
"""

import numpy as np
print("numpy version:", np.__version__)
import os
import io
import re
print("re version:", re.__version__)
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from datetime import datetime
import xbatcher as xb
print("xbatcher version:", xb.__version__)

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
print("Keras version:", keras.__version__)

import xarray as xr
print("xarray version:", xr.__version__)

import logging

import cnn
import preprocess_data

import argparse

from timeit import default_timer as timer

#TODO can we use decorators here e.g.
# https://www.thepythoncodingstack.com/p/demystifying-python-decorators

def parse_args():
    parser = argparse.ArgumentParser(description="Train a NN model on preprocessed data.")
    # filenames and directories
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the preprocessed data files.")
    parser.add_argument("--data_filenames", type=str, required=True, help="List of filenames to load the data from.")
    parser.add_argument("--domain", type=str, default=None, help="Domain identifier for data files.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save/load the trained model.")
    parser.add_argument("--model_filename", type=str, default=None, help="Name of the model file to load.")
    parser.add_argument("--model_save_filename", type=str, default=None, help="Name of the model file to save to.")
    # data slicing #! hard coded for now
    parser.add_argument("--x_c_slice", type=str, default="3:37", help="Slice for x_c dimension, e.g. '3:37'")
    parser.add_argument("--y_c_slice", type=str, default="0:34", help="Slice for y_c dimension, e.g. '0:34'")
    parser.add_argument("--t_slice", type=str, default="0:300", help="Slice for t dimension, e.g. '0:300'") #! not needed
    #// parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save/load model checkpoints.")
    #// parser.add_argument("--checkpoint_filename", type=str, default=None, help="Name of chekcpoint file - model weights.")
    # model features and parameters
    parser.add_argument("--features", type=str, required=True, help="List of feature names to use for training.")
    parser.add_argument("--target", type=str, required=True, help="Target.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train the model.")
    parser.add_argument("--kernels", type=str, default=None, help="Size of the convolutional kernel.")
    parser.add_argument("--padding", type=str, default=None, help="Convolutional padding.")
    parser.add_argument("--filters", type=str, default=None, help="Number of filters in the convolutional layers.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training.")
    parser.add_argument("--k_folds", type=int, default=1, help="Number of folds for k-fold cross validation (1 = no CV).")  
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer. \
                        If --use_learning_rate_scheduler is set, this is the initial learning rate.")
    parser.add_argument("--use_learning_rate_scheduler", action="store_true", \
                         help="Flag to indicate whether to use a learning rate scheduler. \
                            This is exponential decay.") 
    parser.add_argument("--dropout_rate", type=float, default=None, help="Dropout rate for the dropout layers.")
    
    # early stopping
    parser.add_argument("--early_stopping", action="store_true", \
                        help="Flag to indicate whether to use early stopping during training.")

    # key flags associated with training, evaluation, prediction
    parser.add_argument("--train", action="store_true", help="Flag to indicate whether to train the model.")
    parser.add_argument("--evaluate", action="store_true", help="Flag to indicate whether to evaluate the model.")
    parser.add_argument("--predict", action="store_true", help="Flag to indicate whether to predict using the model.")
    #TODO pickup will not work when using learning rate scheduler. Adapt this.
    parser.add_argument("--pickup", action="store_true", help="Flag to indicate whether to pick up training.")
    parser.add_argument("--verbose", action="store_true", help="Verbose output during training.")
    parser.add_argument("--local_norm", action="store_true", help="Normalize training data within local domain.")
    parser.add_argument("--global_norm", action="store_true", help="Normalize training data over all domains.")
    # logging and monitoring
    # parser.add_argument("--tensboard_log_dir", type=str, default=None, help="Directory to save TensorBoard logs.")
    parser.add_argument("--tensboard_log_dir", type=str, default=None, help="Name of the TensorBoard log directory.")
    # parser.add_argument("--verbose_output_dir", type=str, default=None, help="Directory to save verbose output logs.")
    parser.add_argument("--verbose_output_filename", type=str, default=None, help="Name of the verbose output log file.")
    return parser.parse_args()

#TODO add the functions into a separate module.
# def scheduler(epoch, lr):
#     '''
#         Learning rate scheduler function.
#         Decays the learning rate exponentially after 20 epochs.
#     '''
#     if epoch < 20:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)


def train_model(scenario, ds_training, ds_validation,
                epochs=200, 
                batch_size=30,
                mask=None, 
                **kwargs,
                ):
    
    '''
        Some description.
    '''

    # some defaults for optional keyword arguments
    opt_dic = {"learning_rate": None, # default value set earlier
               "use_learning_rate_scheduler": False,
               "dropout_rate": 0.2,
               "use_ema": False,
               "ema_momentum": 0.99,
               "verbose": False,
               "pickup": False,
               "current_period": None,
               "model_dir": "./",
               "model_filename": None,
               "tensboard_log_dir": None, #TODO this needs a default directory
               "early_stopping": False,
    }

    # overwrite the options by cycling through the input dictionary
    for key in kwargs:
        opt_dic[key] = kwargs[key]

    # Detect y_c and x_c dimension sizes from dataset
    y_c_size = ds_training.dims['y_c']
    x_c_size = ds_training.dims['x_c']
    # t_size = ds_training.dims['t']

    # Set up batches
    bgen_training = xb.BatchGenerator(
        ds_training,
        input_dims={'y_c': y_c_size, 'x_c': x_c_size},
        batch_dims={'sample': batch_size},
    )

    # Set up batches
    bgen_validation = xb.BatchGenerator(
        ds_validation,
        input_dims={'y_c': y_c_size, 'x_c': x_c_size},
        batch_dims={'sample': batch_size},
    )

    # Create the generator objects
    train_generator = _keras_data_generator(bgen_training, scenario)
    val_generator = _keras_data_generator(bgen_validation, scenario)

    # Train the model using the generators
    validation_steps = len(ds_validation.sample) // batch_size
    steps_per_epoch = len(ds_training.sample) // batch_size

    # set a learning rate scheduler
    #TODO make this a user option with kwargs
    if opt_dic["use_learning_rate_scheduler"]:
        lr_option = keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=opt_dic["learning_rate"],
                    decay_steps=steps_per_epoch * 30,
                    decay_rate=0.9
                    )
    else:
        lr_option = opt_dic["learning_rate"]

    if opt_dic["pickup"]:
        
        # Load a pre-trained model and continue training
        if opt_dic["verbose"]:
            logger.info('Loading pre-trained model: %s',
                         opt_dic["model_filename"])
        model = keras.saving.load_model(
                    opt_dic["model_dir"] + opt_dic["model_filename"],
                    compile=True,
                    )

        if opt_dic["verbose"]:
            stream = io.StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            logger.info('Model summary is:\n%s', stream.getvalue())

        # get datetime from cnn filename
        # and update kwargs
        fn = opt_dic["model_filename"]
        opt_dic["current_period"] = re.split(r'[_.]+', fn)[1]

    else:
        # Build the CNN
        #* could make this into a function e.g. 
        #* https://stackoverflow.com/questions/41665799/keras-model-summary-object-to-string
        input_shape = (x_c_size,y_c_size,len(scenario.input_var))
        model = cnn.CNN(scenario, input_shape, dropout_rate=opt_dic["dropout_rate"])
        model(tf.keras.Input(shape=input_shape))
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))

        logger.info('Model summary is:\n%s', stream.getvalue())


        # Compile the CNN
        #TODO add logicals with learning_rate_scheduler
        model.compile(optimizer=keras.optimizers.Adam(
            learning_rate=lr_option,
            use_ema=opt_dic["use_ema"], 
            ema_momentum=opt_dic["ema_momentum"],
            ),
            loss=cnn.MaskedMSELoss(mask=mask), # if mask is None, no masking is applied
            run_eagerly=False,
        ) # 

    # Set a log directory for Tensorboard monitoring
    if opt_dic["tensboard_log_dir"] is None:
        log_dir = 'logs/fit/' + opt_dic["current_period"]
        tensorboard_callback = ( tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                            histogram_freq=1, write_images=True) )
        
    else:
        log_dir = opt_dic["tensboard_log_dir"]
        tensorboard_callback = ( 
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, 
                histogram_freq=1, 
                write_images=True,
                update_freq='epoch',) 
            )
    
    callbacks = [tensorboard_callback]

    # TODO set up checkpoint callback to save model weights during training

    # Early stopping callback
    if opt_dic["early_stopping"]:
        earlystopping_callback = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=1e-3,
            start_from_epoch=20,
        )
        callbacks.append(earlystopping_callback)
    
    # learning_rate_callback = keras.callbacks.LearningRateScheduler(
    #             scheduler,
    #             )

    if opt_dic["verbose"]:    
        logger.info('Running model.fit()...')

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    if opt_dic["verbose"]:    
        logger.info('Training loss history: %s', loss)
        logger.info('Validation loss history: %s', val_loss)

    return model

def evaluate_model(scenario, ds_test,
                   model_fn, batch_size=20,
                   **kwargs,
                ):
    
    # TODO do i need to set up batched similar to train?
    
    # some defaults for optional keyword arguments
    opt_dic = {"verbose": False,
               "tensboard_log_dir": None,
               "current_period": None,
               }

    # overwrite the options by cycling through the input dictionary
    for key in kwargs:
        opt_dic[key] = kwargs[key]
    
    model = keras.saving.load_model(
                    model_fn,
                    compile=True,
                    ) #

    if opt_dic["verbose"]:
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        logger.info('Model summary is:\n%s', stream.getvalue())

    # # retrieve input and taget variables
    # batch_input  = [ds_test[x] for x in scenario.input_var]
    # batch_target  = [ds_test[x] for x in scenario.target]
    # # adds an additional dimension for tf readability
    # batch_input  = xr.merge(batch_input).to_array('var').transpose(...,'var') # channels
    # batch_target = xr.merge(batch_target).to_array('var').transpose(...,'var')

    # Set a log directory
    # Set a log directory for Tensorboard monitoring
    if opt_dic["tensboard_log_dir"] is None:
        log_dir = opt_dic["current_period"]
        tensorboard_callback = ( tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                            histogram_freq=1, write_images=True) )
        
    else:
        log_dir = opt_dic["tensboard_log_dir"]
        tensorboard_callback = ( tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                            histogram_freq=1, write_images=True) )

    if opt_dic["verbose"]:
        logger.info('Evaluating...')

    # Detect y_c and x_c dimension sizes from dataset
    y_c_size = ds_test.dims['y_c']
    x_c_size = ds_test.dims['x_c']
    sample_size = ds_test.dims['sample']

    # Set up batches
    bgen = xb.BatchGenerator(
        ds_test,
        input_dims={'y_c': y_c_size, 'x_c': x_c_size},
        batch_dims={'sample': batch_size},
    )

    batch_idx = 1
    loss=np.zeros(round(sample_size/batch_size))
    for batch in bgen:
        # retrieve input and taget variables
        batch_input  = [batch[x] for x in scenario.input_var]
        batch_target  = [batch[x] for x in scenario.target]
        #* adds an additional dimension for tf readability
        batch_input  = xr.merge(batch_input).to_array('var').transpose(...,'var') #* channels
        batch_target = xr.merge(batch_target).to_array('var').transpose(...,'var')

        loss[batch_idx-1] = model.evaluate(batch_input.to_numpy(), 
                        batch_target.to_numpy(), 
                        batch_size=batch_size, 
                        callbacks=[tensorboard_callback],
                        )
        
        if opt_dic["verbose"]:    
            logger.info('End of batch: %s', batch_idx)

        batch_idx+=1

    return loss


def predict_model():
    # some code 
    pass

def _keras_data_generator(bgen, sc):
    """
    A wrapper generator that takes xbatcher generator (bgen)
    and yields Keras-compatible (inputs, targets) tuples.
    """
    while True: # infinite generator for multiple epochs
        for batch in bgen:
            # Your exact preprocessing logic
            batch_input = [batch[x] for x in sc.input_var]
            batch_target = [batch[x] for x in sc.target]
            
            batch_input = xr.merge(batch_input).\
                to_array('var').transpose(...,'var')
            batch_target = xr.merge(batch_target).\
                to_array('var').transpose(...,'var')
    
            # Yield the final NumPy arrays
            yield (batch_input.to_numpy(), batch_target.to_numpy())

# ------------------- some smaller functions -------------------
def parse_slice(slice_str):
    start, end = map(int, slice_str.split(":"))
    return slice(start, end)

# ------------------- main code -------------------
if __name__ == "__main__": # executed when run as a script, not when imported as a module

    args = parse_args()

    current_period = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Set up logging
    if args.verbose:
        logger = logging.getLogger(__name__)
        if args.verbose_output_filename is None:
            
            logging.basicConfig(
                filename=f'verbose_output_{current_period}.log', 
                format="%(asctime)s %(levelname)s %(message)s",
                level=logging.INFO, 
                filemode='w'
            )

        if args.verbose_output_filename is not None:

            logging.basicConfig(
                filename=args.verbose_output_filename, 
                format="%(asctime)s %(levelname)s %(message)s",
                level=logging.INFO, 
                filemode='w'
            )

    # GPU Configuration
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if args.verbose:
                logger.info(f"GPU(s) detected: {len(gpus)}")
        except RuntimeError as e:
            if args.verbose:
                logger.error(f"GPU configuration error: {e}")
    else:
        if args.verbose:
            logger.info("No GPUs detected. Training will use CPU.")


    #todo do some preliminary checks, raise exceptions / errors and exit if necessary
    #todo do these checks first so the programme exits swiflty

    # Checks if either local or global normalization is specified
    # else fail
    if not (args.local_norm or args.global_norm):
        raise ValueError('Please specify either \
                         --local_norm or --global_norm \
                         for data normalization.') 

    # Load the data
    if args.verbose:
        logger.info('Loading data from directory: %s', args.data_dir)
        #//logger.info('Filenames: %s', args.data_filenames)

    directory = args.data_dir
    model_directory = args.model_dir

    domain = args.domain.split()
    if args.verbose:
        logger.info('Domain(s) are: %s', domain)

    filenames = args.data_filenames.split()
    if args.verbose:
        logger.info('Filenames are: %s', filenames)

    features = args.features.split()
    if args.verbose:
        logger.info('Features are: %s', features)

    target = [args.target]
    if args.verbose:
        logger.info('Target is: %s', target)

    if args.filters is not None:
        filters = [int(f) for f in args.filters.split()]
        if args.verbose:
            logger.info('Filters are: %s', filters)
    else:
        raise ValueError('Please provide a list of filter' \
            'sizes for the convolutional layers')

    if args.kernels is not None:
        kernels = [tuple(map(int,k.strip('()').split(','))) \
                    for k in args.kernels.split()]
        if args.verbose:
            logger.info('Kernels are: %s', kernels)
    else:
        raise ValueError('Please provide a list of kernel sizes' \
            'for the convolutional layers')
    
    if args.padding is not None:
        pads = [tuple(map(int,k.strip('()').split(','))) \
                    for k in args.padding.split()]
        if args.verbose:
            logger.info('Padding are: %s', pads)
    else:
        raise ValueError('Please provide a list of padding sizes' \
            'for the convolutional layers')
    
    # loss = args.loss_function
    

    scenario = cnn.Scenario(features,
                            target,
                            filters, 
                            kernels, 
                            pads, 
                            name = 'testing'
    )
    if args.verbose:
        logger.info('Scenario created: %s', scenario)

    # sc = copy.deepcopy(scenario)

    start = timer()
    # local normalization
    if args.local_norm:
        if args.verbose:
            logger.info('Using local normalization for data preprocessing.')

        ds, sc = preprocess_data.open_and_process_data(
                            scenario, 
                            directory, 
                            filenames, 
                            domain,
                            )
        
    # global normalization
    if args.global_norm:
        if args.verbose:
            logger.info('Using global normalization for data preprocessing.')

        ds, mask, scenario = preprocess_data.open_and_combine_data(
                            scenario, 
                            directory, 
                            filenames[:-1], 
                            filenames[-1], 
                            domain,
                            )
        
        processor = preprocess_data.data_preparation(
                            scenario, 
                            dataset=ds,
                            mask=mask,
                            parallel=False,
                            )
        ds = processor()

    end = timer()
    if args.verbose:
        logger.info('Data loading and preprocessing time: %.2f seconds', end - start)
    if args.verbose:
        logger.info('Dataset loaded and preprocessed: %s', ds)

    # if args.verbose:
    #     logger.info('Same input scenario after processing: %s', scenario)

    if args.verbose:
        logger.info('Scenario updated after processing: %s', scenario)

    # # get mask
    # mask = xr.open_dataset(directory + filenames[-1])
    # tmask = mask.tmask.isel(x_c=slice(0, 39), y_c=slice(0,39), t=0, nav_lev=0)
    # tmask = tmask.expand_dims({'batch':args.batch_size, 'var': 1})
    # tmask = tmask.transpose('batch', 'y_c', 'x_c', 'var').to_numpy()

    # train the model
    if args.train:

        #todo retrain or continue training from a pre-trained model
        #* load in weights.

        # parse specific arguments
        if args.epochs is not None:
            epochs = args.epochs
        else:
            raise ValueError("You must supply the number \
                              of epochs for training using --epochs.")
        
        if args.batch_size is not None:
            batch_size = args.batch_size
        else:
            raise ValueError("You must supply the batch size \
                              for training using --batch_size.")

        #! do not need as deault is set in function
        # if args.learning_rate is not None:
        #     learning_rate = args.learning_rate
        # else:
        #     learning_rate = 0.001 # default value
        
        # if user specified pickup flag, then model_filename must be supplied
        if args.pickup:
            if args.model_filename is None:
                raise ValueError("You must supply the model filename \
                                  for picking up training using --model_filename.")
        
        # # Extract the training data sample
        # x_c_slice = parse_slice(args.x_c_slice)
        # y_c_slice = parse_slice(args.y_c_slice)
        # t_slice = parse_slice(args.t_slice)

        #! hard coded data split for now
        # ----------------------------
        # 2. Define split sizes
        # ----------------------------
        n_test = 359     # one less due to eke shift
        n_val  = 360    # 60 days before test
        train_stride = 1  # every day

        nt = ds.sizes["t"]

        # ----------------------------
        # 3. Create time indices
        # ----------------------------
        test_idx = np.arange(nt - n_test, nt)
        val_idx  = np.arange(nt - n_test - n_val, nt - n_test)
        train_idx_full = np.arange(1, nt - n_test - n_val) # start from day 2 as tendency starts from day 2

        # ----------------------------
        # 4. Subsample training every 1st day
        # ----------------------------
        train_idx = train_idx_full[::train_stride]

        # ----------------------------
        # 5. Create split datasets
        # ----------------------------
        ds_train = ds.isel(t=train_idx)
        ds_val   = ds.isel(t=val_idx)
        ds_test  = ds.isel(t=test_idx)

        ds_flat = ds_train.stack(sample=('r', 't')) 
        # shuffle the samples
        n_samples = ds_flat.sample.size
        perm = np.random.permutation(n_samples)
        ds_flat = ds_flat.isel(sample=perm)

        #* commented out below method for subsampling data for above
        # # select time bins for training and testing data
        # time_bins = [1, 2, 4, 5, 7, 8, 10, 11] #* Hard coded for now
        # # subset the data
        # subset_training = ds.sel(t=ds['t.month'].isin(time_bins))
        # # flatten r and t dimensions into one sample dimension
        # ds_flat = subset_training.stack(sample=('r', 't')) 
        # # shuffle the samples
        # n_samples = ds_flat.sample.size
        # perm = np.random.permutation(n_samples)
        # ds_flat = ds_flat.isel(sample=perm)

        if args.verbose:
            logger.info('Dataset for training is: %s', ds_flat)

        # ds_training = ds.isel(
        #     x_c=x_c_slice,
        #     y_c=y_c_slice,
        #     t=t_slice
        # )

        # cross validation
        if args.k_folds == 1:

            ds_validation = ds_val.stack(sample=('r', 't'))
            if args.verbose:
                logger.info('Dataset for validation is: %s', ds_validation)

            #* commented out below method for subsampling data for validation
            # # select validation data
            # time_bins = [6, 12] # hard coded for summer and winter
            # subset_validation = ds.sel(t=ds['t.month'].isin(time_bins))
            # ds_val_flat = subset_validation.stack(sample=('r', 't'))
            # if args.verbose:
            #     logger.info('Dataset for validation is: %s', ds_val_flat)
            # ds_validation = ds_val_flat.isel(
            #     sample=np.random.permutation(ds_val_flat.sample.size)
            #     )
        
            # train the model
            #TODO when picking up, need to get learning rate
            if args.pickup:
                kwargs = {"use_ema": True,
                        "learning_rate": args.learning_rate,
                        "dropout_rate": args.dropout_rate,
                        "use_learning_rate_scheduler": args.use_learning_rate_scheduler,
                        "verbose": args.verbose,
                        "pickup": args.pickup,
                        "model_dir": model_directory,
                        "model_filename": args.model_filename,
                        "tensboard_log_dir": args.tensboard_log_dir,
                        "early_stopping": args.early_stopping,
                        }
                model = train_model(scenario, 
                                    ds_flat,
                                    ds_validation,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    mask=None,
                                    **kwargs,
                )

            else:
                kwargs = {"use_ema": True,
                        "dropout_rate": args.dropout_rate,
                        "learning_rate": args.learning_rate,
                        "use_learning_rate_scheduler": args.use_learning_rate_scheduler,
                        "verbose": args.verbose,
                        "pickup": args.pickup,
                        "current_period": current_period,
                        "tensboard_log_dir": args.tensboard_log_dir,
                        "early_stopping": args.early_stopping,
                        }
                model = train_model(scenario, 
                                    ds_flat,
                                    ds_validation,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    mask=None,
                                    **kwargs,
                )

            # Save the model
            if args.pickup:
                if args.model_save_filename:
                    fn = args.model_save_filename
                else:
                    #* overwrite
                    fn = args.model_filename
            elif args.model_save_filename:
                fn = args.model_save_filename
            else:
                fn = f'cnn_{current_period}.keras'
                
            logger.info('Saving model: %s', fn)
            model.save(model_directory + fn)

        #! k-fold cross validation not tested with new dataset split
        else: # k-fold>1
            # prepare indices for k-fold CV over 'sample' axis
            indices = np.arange(n_samples)
            folds = np.array_split(indices, args.k_folds)

            saved_models = []
            for fold_idx in range(args.k_folds):
                val_idx = folds[fold_idx]
                # concat remaining folds for training
                train_idx = np.concatenate(
                    [f for i, f in enumerate(folds) if i != fold_idx]
                )

                ds_training_fold = ds_flat.isel(sample=train_idx)
                ds_validation_fold = ds_flat.isel(sample=val_idx)

                kwargs = {"use_ema": True,
                        "dropout_rate": args.dropout_rate,
                        "learning_rate": args.learning_rate,
                        "use_learning_rate_scheduler": 
                        args.use_learning_rate_scheduler,
                        "verbose": args.verbose,
                        "pickup": args.pickup,
                        "current_period": f"{current_period}_fold{fold_idx+1}",
                        "tensboard_log_dir": args.tensboard_log_dir,
                        "early_stopping": args.early_stopping,
                        }
                model = train_model(scenario, 
                                    ds_training_fold,
                                    ds_validation_fold,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    mask=None,
                                    **kwargs,
                )

                if args.model_save_filename:
                    fn = args.model_save_filename
                else:
                    fn = f'cnn_{current_period}_fold{fold_idx+1}.keras'
                    
                logger.info('Saving model (fold %d): %s', fold_idx+1, fn)
                model.save(model_directory + fn)
                saved_models.append((fn, model))

            if args.verbose:
                logger.info('Saved models: %s', saved_models)

    # evaluate the model
    if args.evaluate:

        if args.batch_size is not None:
            batch_size = args.batch_size
        else:
            raise ValueError("You must supply the batch size \
                              for evaluation using --batch_size.")
        
        # Load the model
        if args.model_filename is not None:
            fn = args.model_filename
        else:
            raise ValueError("You must supply the model filename \
                              for evaluation using --model_filename.")
        
        # select validation data
        time_bins = [6, 12] # hard coded for summer and winter
        subset_validation = ds.sel(t=ds['t.month'].isin(time_bins))
        ds_val_flat = subset_validation.stack(sample=('r', 't'))
        ds_validation = ds_val_flat.isel(
            sample=np.random.permutation(ds_val_flat.sample.size)
            )

        if args.verbose:
            logger.info('dataset for evaluation is: %s', ds_validation)
        
        model_fn = model_directory + fn
        
        kwargs = {"verbose": args.verbose,
                  "tensboard_log_dir": args.tensboard_log_dir,
                 }
        loss = evaluate_model(scenario, 
                              ds_validation,
                              model_fn, 
                              batch_size=batch_size,
                              **kwargs,
                            )

        if args.verbose:
            logger.info('Evaluation loss: %s', loss)

        