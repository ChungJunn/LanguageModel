DATA=$2
MAX_LENGTH=150

DATA_NUM=2
DIM_WEMB=300
DIM_ENC=300

if [ $DATA == 'wiki2' ]
then
    DATA_DIR=$HOME'/data/wikitext-'$DATA_NUM'-raw'
    TRAIN_FILE=$DATA_DIR'/wiki.train.raw'
    VALID_FILE=$DATA_DIR'/wiki.valid.raw'
    TEST_FILE=$DATA_DIR'/wiki.test.raw'
    DICT1=$TRAIN_FILE'.voc.pkl'
elif [ $DATA == 'ptb' ]
then
    DATA_DIR=$HOME'/data/ptb'
    TRAIN_FILE=$DATA_DIR'/ptb.train.txt'
    VALID_FILE=$DATA_DIR'/ptb.valid.txt'
    TEST_FILE=$DATA_DIR'/ptb.test.txt'
    DICT1=$TRAIN_FILE'.pkl'
fi

###TODO: must change######
LR=0.01
OPTIMIZER='adam'
PATIENCE=30
LOSS_REDUCTION='con'
DROPOUT_P=0.5
NUM_LAYERS=1
BATCH_SIZE=64
LR_DECAY=10

PRINT_EVERY=100
VALID_EVERY=$3
VALID_START=1000

INIT='cjlee/LanguageModel'
NAME=$DATA'-adam-lr'
TAG='0-01'
##########################

SAVE_DIR='./result'

CUDA_VISIBLE_DEVICES=$1 python3 lm_run.py --train=1 \
        --train_data_file=$TRAIN_FILE --valid_data_file=$VALID_FILE --test_data_file=$TEST_FILE \
        --dim_wemb=$DIM_WEMB --dim_enc=$DIM_ENC --save_dir=$SAVE_DIR --dropout_p=$DROPOUT_P\
        --data_dict=$DICT1 --max_length=$MAX_LENGTH --batch_size=$BATCH_SIZE\
        --init=$INIT --name=$NAME --tag=$TAG --loss_reduction=$LOSS_REDUCTION \
        --learning_rate=$LR --optimizer=$OPTIMIZER --patience=$PATIENCE --num_layers=$NUM_LAYERS \
        --print_every=$PRINT_EVERY --valid_every=$VALID_EVERY --val_start=$VALID_START --lr_decay=$LR_DECAY
