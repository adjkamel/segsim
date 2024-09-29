
from train_32 import train_32
from train_64 import train_64
from train_128 import train_128
from train_256 import train_256
from train_fusion import train_fusion
from eval_unsup import eval_unsup
from arguments import args



if args.algo == 'sup':

    if args.model_type == 'fusion':
        train_fusion()

    if args.model_type == '32':
        train_32()

    if args.model_type == '64':
        train_64()
     
    if args.model_type == '128':
        train_128()

    if args.model_type == '256':
        train_256()

if args.algo == 'unsup':

    eval_unsup()

        

