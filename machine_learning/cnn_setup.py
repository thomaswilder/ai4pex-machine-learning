#!/usr/bin/env python3

'''
    Description: 
'''

import cnn
import preprocess_data

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

    logger.info(
        f"Data loading and processing complete. \
                 Dataset: {ds}, Scenario: {sc}"
    )

    return ds, sc

def get_data_split(): pass