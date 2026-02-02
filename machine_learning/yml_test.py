#!/usr/bin/python3

import argparse
import yaml


def parse_args():

    parser = argparse.ArgumentParser(description="Train a NN model on preprocessed data.")

    # NEW: YAML config
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")

    # filenames and directories
    parser.add_argument("--data_dir", type=str, help="Directory containing the preprocessed data files.")
    parser.add_argument("--data_filenames", type=str, help="List of filenames to load the data from.")
    # parser.add_argument("--domain", type=str, default=None, help="Domain identifier for data files.")
    parser.add_argument("--model_dir", type=str, help="Directory to save/load the trained model.")
    # parser.add_argument("--model_filename", type=str, default=None, help="Name of the model file to load.")
    # parser.add_argument("--model_save_filename", type=str, default=None, help="Name of the model file to save to.")
    # # data slicing #! hard coded for now
    # parser.add_argument("--x_c_slice", type=str, default="3:37", help="Slice for x_c dimension, e.g. '3:37'")
    # parser.add_argument("--y_c_slice", type=str, default="0:34", help="Slice for y_c dimension, e.g. '0:34'")
    # parser.add_argument("--t_slice", type=str, default="0:300", help="Slice for t dimension, e.g. '0:300'") #! not needed
    # #// parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save/load model checkpoints.")
    # #// parser.add_argument("--checkpoint_filename", type=str, default=None, help="Name of chekcpoint file - model weights.")
    # # model features and parameters
    # parser.add_argument("--features", type=str, required=True, help="List of feature names to use for training.")
    # parser.add_argument("--target", type=str, required=True, help="Target.")
    # parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train the model.")
    # parser.add_argument("--kernels", type=str, default=None, help="Size of the convolutional kernel.")
    # parser.add_argument("--padding", type=str, default=None, help="Convolutional padding.")
    # parser.add_argument("--filters", type=str, default=None, help="Number of filters in the convolutional layers.")
    # parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training.")
    # parser.add_argument("--k_folds", type=int, default=1, help="Number of folds for k-fold cross validation (1 = no CV).")  
    # parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer. \
    #                     If --use_learning_rate_scheduler is set, this is the initial learning rate.")
    # parser.add_argument("--use_learning_rate_scheduler", action="store_true", \
    #                         help="Flag to indicate whether to use a learning rate scheduler. \
    #                         This is exponential decay.") 
    parser.add_argument("--dropout_rate", type=float, default=None, help="Dropout rate for the dropout layers.")
    # # # key flags associated with training, evaluation, prediction
    # parser.add_argument("--train", action="store_true", help="Flag to indicate whether to train the model.")
    # parser.add_argument("--evaluate", action="store_true", help="Flag to indicate whether to evaluate the model.")
    # parser.add_argument("--predict", action="store_true", help="Flag to indicate whether to predict using the model.")
    # # #TODO pickup will not work when using learning rate scheduler. Adapt this.
    # parser.add_argument("--pickup", action="store_true", help="Flag to indicate whether to pick up training.")
    # parser.add_argument("--verbose", action="store_true", help="Verbose output during training.")
    # parser.add_argument("--local_norm", action="store_true", help="Normalize training data within local domain.")
    # parser.add_argument("--global_norm", action="store_true", help="Normalize training data over all domains.")
    # # logging and monitoring
    # # parser.add_argument("--tensboard_log_dir", type=str, default=None, help="Directory to save TensorBoard logs.")
    # parser.add_argument("--tensboard_log_dir", type=str, default=None, help="Name of the TensorBoard log directory.")
    # # parser.add_argument("--verbose_output_dir", type=str, default=None, help="Directory to save verbose output logs.")
    # parser.add_argument("--verbose_output_filename", type=str, default=None, help="Name of the verbose output log file.")

    pre_args, remaining = parser.parse_known_args()

    print("Pre args:", pre_args)
    print("Remaining args:", remaining)

    cfg = {}
    if pre_args.config:
        cfg = yaml.safe_load(open(pre_args.config, 'r'))

        valid_dests = set()
        for action in parser._actions:
            valid_dests.add(action.dest)

        print("Valid dests:", valid_dests)

        kept = {}
        for k, v in cfg.items():
            if k in valid_dests:
                kept[k] = v

        print("Kept:", kept)

        parser.set_defaults(**kept)

    args = parser.parse_args(remaining)
    # args.config = pre_args.config

    return args

args = parse_args()

print(args)

print(args.data_dir)