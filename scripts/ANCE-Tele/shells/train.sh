# Current train episode. Used for deciding number of train passages.
EPISODE=$1
# Comma-seperated index of CUDA device to use.
CUDA_TO_USE=$2
# Path to the starting model.
PREVIOUS_MODEL_DIR=$3
# Path to save the trained model.
CURRENT_MODEL_DIR=$4
# Path to the directory for the train files, or path to a single train file (.jsonl)
TRAIN_FILE=$5
# Path to save tensorboard logs.
LOG_PATH=$6
# Path to store the outputs to the terminal (useful for debugging)
SCRIPT_LOG_FILE=$7
# If set to anything other than 0, the training would be cutted off in 20000 steps. Used to free computing resources.
CUTOFF=$8
PORT=$9
CUDAs=(${CUDA_TO_USE//,/ })
CUDA_NUM=${#CUDAs[@]}
CUTOFF=${CUTOFF:-0}
PORT=${PORT:-$RANDOM}
# number of train passages differ between epochs
echo "Beginning Training episode ${EPISODE}..."
if [ $EPISODE -eq 1 ]; then
N_PASSAGES=16
else
N_PASSAGES=32
fi
echo "Learning $[ N_PASSAGES-1 ] negatives per query."
# if input is a single file, we pass it as a file
if echo $TRAIN_FILE | grep -q -E "\.json(l?)$" ; then
PATH_OR_DIR="train_path"
else
PATH_OR_DIR="train_dir"
fi
# Set whether to distribute depending on number of device provided
if [ $CUDA_NUM -gt 1 ]; then
echo "Distributed training on CUDA ${CUDA_TO_USE}, managing on port ${PORT}"
CUDA_VISIBLE_DEVICES=$CUDA_TO_USE python -m torch.distributed.launch --nproc_per_node=2 --master_port ${PORT} \
    -m openmatch.driver.train_dr  \
    --output_dir $CURRENT_MODEL_DIR  \
    --model_name_or_path $PREVIOUS_MODEL_DIR  \
    --do_train  \
    --save_steps 20000  \
    --$PATH_OR_DIR ${TRAIN_FILE}  \
    --fp16  \
    --per_device_train_batch_size $[8/$CUDA_NUM]  \
    --negatives_x_device \
    --train_n_passages $N_PASSAGES  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 3  \
    --logging_dir ${LOG_PATH}  \
    --use_mapping_dataset \
    --dataloader_drop_last \
&> ${SCRIPT_LOG_FILE} &
else
echo "Training on CUDA ${CUDA_TO_USE}"
CUDA_VISIBLE_DEVICES=$CUDA_TO_USE python -m openmatch.driver.train_dr  \
    --output_dir $CURRENT_MODEL_DIR  \
    --model_name_or_path $PREVIOUS_MODEL_DIR  \
    --do_train  \
    --save_steps 20000  \
    --$PATH_OR_DIR ${TRAIN_FILE}  \
    --fp16  \
    --per_device_train_batch_size $[8/$CUDA_NUM]  \
    --train_n_passages $N_PASSAGES  \
    --learning_rate 5e-6  \
    --q_max_len 32  \
    --p_max_len 128  \
    --num_train_epochs 3  \
    --logging_dir ${LOG_PATH} \
    --use_mapping_dataset \
    --dataloader_drop_last \
&> ${SCRIPT_LOG_FILE} &
fi

if [ $CUTOFF -gt 0 ]; then
    echo "Waiting for checkpoint-20000 of train episode ${EPISODE}..."
    until test -f $CURRENT_MODEL_DIR/checkpoint-20000/pytorch_model.bin; do
    sleep 5m
    done
    echo "Wait for the model dumping..."
    sleep 1m # Wait until the model is fully dumped
    kill $!
else
    echo "Waiting for training to complete..."
    until test -f $CURRENT_MODEL_DIR/pytorch_model.bin; do
    sleep 5m
    done
fi