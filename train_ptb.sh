BPE=1

RNN=$2
RNN_FF=$3
MAX_LENGTH=$4

DATA_NUM=2
DIM_WEMB=200
DIM_ENC=300

DATA_DIR=$HOME'/data/ptb'
TRAIN_FILE=$DATA_DIR'/ptb.train.txt'
VALID_FILE=$DATA_DIR'/ptb.valid.txt'
TEST_FILE=$DATA_DIR'/ptb.test.txt'

###TODO: must change######
LR=0.0002
OPTIMIZER='adam'
PATIENCE=5
LOSS_REDUCTION='con'

INIT='cjlee/LanguageModel'
NAME='PTB-LearningRate-Adam'
TAG='0-0002'
##########################

DICT1=$TRAIN_FILE'.pkl'
SAVE_DIR='./results'

MODEL_FILE='lm.ptb.'$RNN'.rnn_ff'$RNN_FF'.'$MAX_LENGTH'.'$DIM_WEMB'.'$DIM_ENC'.gpu'$1

CUDA_VISIBLE_DEVICES=$1 python3 lm_run.py --train=1 --rnn_name=$RNN --rnn_ff=$RNN_FF \
        --save_dir=$SAVE_DIR --model_file=$MODEL_FILE \
        --train_data_file=$TRAIN_FILE --valid_data_file=$VALID_FILE --test_data_file=$TEST_FILE \
        --dim_wemb=$DIM_WEMB --dim_enc=$DIM_ENC \
        --data_dict=$DICT1 --max_length=$MAX_LENGTH \
        --init=$INIT --name=$NAME --tag=$TAG --loss_reduction=$LOSS_REDUCTION \
        --learning_rate=$LR --optimizer=$OPTIMIZER --patience=$PATIENCE\
        --print_every=100 --valid_every=1000 --val_start=5
