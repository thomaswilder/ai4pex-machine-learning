#!/usr/bin/env python3

'''
    Description: 


'''


from parsing_args import parse_args
from cnn_setup import setup_scenario, get_data
import logging




if __name__ == "__main__":

    # parsing the arguments
    args = parse_args(None)

    # setting up the logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
                filename=args.verbose_output_filename, 
                format="%(asctime)s %(levelname)s %(message)s",
                level=logging.INFO, 
                filemode='w'
    )

    # # get the scenario for the CNN model
    # scenario = setup_scenario(args, logger)

    # # get the data for the scenario
    # #! only using local normalisation for now
    # ds, sc = get_data(scenario, args, logger)

    print("hey")