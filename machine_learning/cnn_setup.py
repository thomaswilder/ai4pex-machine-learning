#!/usr/bin/env python3

'''
    Description: 
'''

import cnn
import preprocess_data
import numpy as np

def setup_scenario(args, logger=None):

    if logger and args.verbose:
        logger.info(
            f"Setting up scenario with features: {args.features}, \
                     target: {args.target}, filters: {args.filters}, \
                        kernels: {args.kernels}, \
                            padding: {args.padding}"
        )

    scenario = cnn.Scenario(
        input_var=args.features, 
        target=args.target, 
        filters=args.filters, 
        kernels=args.kernels,
        padding=args.padding,
        name=None,
    )

    logger.info(f"Scenario setup complete: {scenario}")

    return scenario

def get_data(scenario, args, logger):

    if logger and args.verbose:
        logger.info(
            f"Getting data with features: {args.features}, \
                     target: {args.target}, data_dir: {args.data_dir}, \
                          data_filenames: {args.data_filenames}, \
                              domain: {args.domain}"
        )

    if args.local_norm:
        ds, sc = preprocess_data.open_and_process_data(
                                scenario, 
                                args.data_dir, 
                                args.data_filenames, 
                                args.domain,
                                )
    elif args.global_norm:
        raise NotImplementedError("Global normalization not implemented yet.")
    elif args.local_norm==None or args.global_norm==None:
        raise ValueError("Must specify either local_norm or global_norm.")

    logger.info(
        f"Data loading and processing complete. \
                 Dataset: {ds}, Scenario: {sc}"
    )

    return ds, sc

def get_data_split(ds, args, logger): 


    #TODO set a default data split if not provided in the config file
    # Set default data split if not provided in the config file
    n_train = args.train_ratio if args.train_ratio is not None else 0.7
    n_val = args.val_ratio if args.val_ratio is not None else 0.15
    n_test = args.test_ratio if args.test_ratio is not None else 0.15

    # set the stride for training samling
    train_stride = args.train_stride if args.train_stride is not None else 1

    if logger and args.verbose:
        logger.info(
            f"Splitting data with train_ratio: {n_train}, \
                     val_ratio: {n_val}, test_ratio: {n_test}, \
                          train_stride: {train_stride}"
        )

    # Validate that ratios sum to 1.0
    total_ratio = n_train + n_val + n_test
    if not abs(total_ratio - 1.0) < 1e-6:
        raise ValueError(f"Train, val, and test ratios must sum to 1.0, got {total_ratio}")

    # Get total number of time steps in the dataset
    nt = ds.sizes["t"]

    n_train = int(nt * n_train)
    n_val = int(nt * n_val)
    n_test = int(nt * n_test)

    # get indices for each split
    test_idx = np.arange(nt - n_test, nt)
    val_idx  = np.arange(nt - n_test - n_val, nt - n_test)
    train_idx_full = np.arange(0, nt - n_test - n_val)

    train_idx = train_idx_full[::train_stride]

    # get the data splits and return
    if args.train:
        ds_train = ds.isel(t=train_idx)
        ds_val   = ds.isel(t=val_idx)
        logger.info(
            f"Datasets split complete for training. Train set: {ds_train}, Val set: {ds_val}"
        )
        return ds_train, ds_val
    else:
        ds_val   = ds.isel(t=val_idx)
        ds_test  = ds.isel(t=test_idx)
        logger.info(
            f"Datasets split complete. Val set: {ds_val}, Test set: {ds_test}"
        )
        return ds_val, ds_test

def get_data_shuffle(ds, args, logger):
    
    
    