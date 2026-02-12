#!/usr/bin/env python3

'''
    Description: 
'''

import cnn

def setup_scenario(args, logger=None):

    if logger and args.verbose:
        logger.info(f"Setting up scenario with features: {args.features}, \
                     target: {args.target}, filters: {args.filters}, \
                        kernels: {args.kernels}, \
                            padding: {args.padding}")

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

def get_data(): pass

def get_data_split(): pass