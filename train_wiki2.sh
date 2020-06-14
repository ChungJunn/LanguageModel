BPE=1

MAX_LENGTH=$2

DATA_NUM=2
DIM_WEMB=200
DIM_ENC=300

DATA_DIR=$HOME'/data/wikitext-'$DATA_NUM'-raw'
TRAIN_FILE=$DATA_DIR'/wiki.train.raw'
VALID_FILE=$DATA_DIR'/wiki.valid.raw'
TEST_FILE=$DATA_DIR'/wiki.test.raw'

###TODO: must change######
LR=0.0002
OPTIMIZER='adam'
PATIENCE=5
LOSS_REDUCTION='con'
DROPOUT_P=0.5

INIT='cjlee/LanguageModel'
NAME='wiki2-test-dropout'
TAG='dropout0-5'
##########################

DICT1=$TRAIN_FILE'.voc.pkl'
SAVE_DIR='./results'

MODEL_FILE='lm.wiki-'$DATA_NUM'.'$RNN'.rnn_ff'$RNN_FF'.'$MAX_LENGTH'.'$DIM_WEMB'.'$DIM_ENC'.gpu'$1

CUDA_VISIBLE_DEVICES=$1 python3 lm_run.py --train=1 \
        --save_dir=$SAVE_DIR --model_file=$MODEL_FILE \
        --train_data_file=$TRAIN_FILE --valid_data_file=$VALID_FILE --test_data_file=$TEST_FILE \
        --dim_wemb=$DIM_WEMB --dim_enc=$DIM_ENC \
        --data_dict=$DICT1 --max_length=$MAX_LENGTH \
        --init=$INIT --name=$NAME --tag=$TAG --loss_reduction=$LOSS_REDUCTION \
        --learning_rate=$LR --optimizer=$OPTIMIZER --patience=$PATIENCE\
        --print_every=100 --valid_every=1000 --val_start=5
