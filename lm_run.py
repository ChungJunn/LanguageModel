import argparse
import neptune
from lm_main import train_model

parser = argparse.ArgumentParser(description="", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--model_file", type=str, default='')
parser.add_argument("--train_data_file", type=str, default='')
parser.add_argument("--valid_data_file", type=str, default='')
parser.add_argument("--test_data_file", type=str, default='')
parser.add_argument("--data_dict", type=str, default='')
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--emb_act", type=int, default=0)
parser.add_argument("--bleu_script", type=str, default='multi-bleu.perl')
parser.add_argument("--rnn_name", type=str, default='gru')
parser.add_argument("--rnn_ff", type=int, default=0)
parser.add_argument("--optimizer", type=str, default='adam')
parser.add_argument("--grad_clip", type=float, default=0.0)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--dropout_p", type=float, default=0.1)
parser.add_argument("--emb_noise", type=float, default=0.0)
parser.add_argument("--reload", type=int, default=0)
parser.add_argument("--h0_init", type=int, default=0)
parser.add_argument("--dim_enc", type=int, default=0)
parser.add_argument("--dim_wemb", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--valid_every", type=int, default=5000)
parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
parser.add_argument("--train", type=int, default=0)
parser.add_argument("--trans", type=int, default=0)
parser.add_argument("--use_best", type=int, default=0)
parser.add_argument("--beam_width", type=int, default=1)
parser.add_argument("--init", type=str, default='')
parser.add_argument("--name", type=str, default='')
parser.add_argument("--tag", type=str, default='')
parser.add_argument("--val_start", type=int, default='')
parser.add_argument("--patience", type=int, default='')
parser.add_argument("--loss_reduction", type=str, default='')

args = parser.parse_args()
print(args)

params = vars(args)

neptune.init(args.init)
exp = neptune.create_experiment(name=args.name, params=params)
neptune.append_tag(args.tag)

args.exp_id = exp._id
#args.exp_id = 'abc' # for testing  

# training
if args.train:
    print ('Training...')
    with open(args.save_dir+'/'+args.model_file+'.args', 'w') as fp:
        for key in vars(args):
            fp.write(key + ': ' + str(getattr(args, key)) + '\n')
    train_model(args, neptune)

'''
if args.trans:
    print ('Translating...')
    bleu_score = translate_file(args)
    if bleu_score >=0: 
        print ('bleu_score', bleu_score)

print ('Done')
'''
