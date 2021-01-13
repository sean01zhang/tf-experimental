"""Run a training job on Cloud ML Engine for a given use case.
Usage:
    trainer.task --train_data_paths <train_data_paths> --output_dir <outdir> 
                [--batch_size <batch_size>] [--hidden_units <hidden_units>]
Options:
    -h --help     Show this screen.
    --batch_size <batch_size>  Integer value indiciating batch size [default: 150]
    --hidden_units <hidden_units>  CSV seperated integers indicating hidden layer 
    sizes. For a fully connected model.', [default: 100]
"""
from docopt import docopt

import model  # Your model.py file.

if __name__ == '__main__':
    arguments = docopt(__doc__)
    # Assign model variables to commandline arguments
    model.TRAIN_PATHS = arguments['<train_data_paths>']
    model.BATCH_SIZE = int(arguments['--batch_size'])
    model.HIDDEN_UNITS = [int(h)
                          for h in arguments['--hidden_units'].split(',')]
    model.OUTPUT_DIR = arguments['<outdir>']
    # Run the training job
    model.train_and_evaluate()
