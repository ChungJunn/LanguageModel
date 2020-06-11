BPE=1

RNN=$2
RNN_FF=$3
MAX_LENGTH=$4

DATA_NUM=2
DIM_WEMB=200
DIM_ENC=300

#DATA_NUM=103
#DIM_WEMB=500
#DIM_ENC=1000

DATA_DIR=$HOME'/data/wikitext-'$DATA_NUM'-raw'
TRAIN_FILE=$DATA_DIR'/wiki.train.raw'
VALID_FILE=$DATA_DIR'/wiki.valid.raw'
TEST_FILE=$DATA_DIR'/wiki.test.raw'

DICT1=$TRAIN_FILE'.voc.pkl'
SAVE_DIR='./results'

MODEL_FILE='lm.wiki-'$DATA_NUM'.'$RNN'.rnn_ff'$RNN_FF'.'$MAX_LENGTH'.'$DIM_WEMB'.'$DIM_ENC'.gpu'$1

CUDA_VISIBLE_DEVICES=$1 python3 lm_run.py --train=1 --rnn_name=$RNN --rnn_ff=$RNN_FF \
        --save_dir=$SAVE_DIR --model_file=$MODEL_FILE \
        --train_data_file=$TRAIN_FILE --valid_data_file=$VALID_FILE --test_data_file=$TEST_FILE \
        --dim_wemb=$DIM_WEMB --dim_enc=$DIM_ENC \
        --data_dict=$DICT1 \
        --valid_every=500 --max_length=$MAX_LENGTH
        #--print_every=10 --valid_every=100
