#!/usr/bin/env python3

'''
    Description: 


'''


from parsing_args import parse_args
from cnn_setup import setup_scenario
import logging




if __name__ == "__main__":

    args = parse_args(None)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
                filename=args.verbose_output_filename, 
                format="%(asctime)s %(levelname)s %(message)s",
                level=logging.INFO, 
                filemode='w'
    )

    scenario = setup_scenario(args, logger)

    print("hey")