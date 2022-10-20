
# The dir of the starting model.
PLM_DIR=~/datas/OpenMatch-New/models/co-condenser-marco
# Path to openmatch scripts.
OPENMATCH_SCRIPTS_DIR=../..
# The name of model, used for naming directories.
PLM_NAME=co-condenser

# The path to download datasets (Rocketqa-labeled MARCO in this case)
COLLECTION_DIR=~/datas/OpenMatch-New/datasets
# Path to store embedded corpus.
EMBEDDING_DIR=~/datas/OpenMatch-New/embeddings
# Path to store retrieved documents (hard-negative or dev set)
RESULT_DIR=~/datas/OpenMatch-New/retrieved
# Path to store training datasets (datas loaded for trainer)
PROCESSED_DIR=~/datas/OpenMatch-New/train_data
# Path to store the models and checkpoints during training.
MODEL_DIR=~/datas/OpenMatch-New/models
# Path to store logs of training (for tensorboard to recall)
LOG_DIR=~/datas/OpenMatch-New/logs
# Path to store logs of this script (Because training are running in backend)
SCRIPT_LOG_DIR=~/datas/OpenMatch-New/script_log
# Number of CUDA devices to use for each episode.
# The length of this list determines the number of episode to train.
# Negatives will first be mined on that device, then trained on that device.
# If multiple device is defined, distributed processing would be automatically deployed.

CUDA_LIST=("4,7" "4,7" "4,7")


EPISODE_NUM=${#CUDA_LIST[@]}
if [ ! -f "$PLM_DIR/config.json" ]; then
echo "Downloading coCondenser to ${PLM_DIR}..."
PLM_DIR=$PLM_DIR bash ./download_model.sh
fi
if [ ! -f "$COLLECTION_DIR/qrels.dev.restructured.tsv" ] ; then
echo "Downloading MS MARCO to $COLLECTION_DIR..."
COLLECTION_DIR=$COLLECTION_DIR OPENMATCH_SCRIPTS_DIR=$OPENMATCH_SCRIPTS_DIR bash ./download_data.sh 
fi
mkdir -p $SCRIPT_LOG_DIR
for ((i=0; i<$EPISODE_NUM; i++))
do
    if [ $i -eq 0 ]; then
        echo "Starting Episode ${i} Mining..."
        source ./build_negative.sh $i ${CUDA_LIST[$i]} $PLM_DIR \
         ${EMBEDDING_DIR}/epi-${i}.${PLM_NAME}-mine \
         ${RESULT_DIR}/epi-${i}.${PLM_NAME}-mine \
         ${PROCESSED_DIR}/epi-$[$i+1].${PLM_NAME} \
         ${PROCESSED_DIR}/epi-${i}.${PLM_NAME} \
          &> $SCRIPT_LOG_DIR/epi-${i}.build.log && \
        echo "Starting Episode $[$i+1] training..." && \
        source ./train.sh $[$i+1] ${CUDA_LIST[$i]} $PLM_DIR $MODEL_DIR/epi-$[$i+1].${PLM_NAME} \
         $PROCESSED_DIR/epi-$[$i+1].$PLM_NAME/ \
         $LOG_DIR/epi-$[$i+1].$PLM_NAME/ \
         $SCRIPT_LOG_DIR/epi-$[$i+1].train.log \
         $[ $EPISODE_NUM-$i-1 ] \
        
    else
        echo "Starting Episode ${i} Mining..."
        source ./build_negative.sh $i ${CUDA_LIST[$i]} $MODEL_DIR/epi-${i}.${PLM_NAME}-cp20000/ \
         ${EMBEDDING_DIR}/epi-${i}.${PLM_NAME}-mine \
         ${RESULT_DIR}/epi-${i}.${PLM_NAME}-mine \
         ${PROCESSED_DIR}/epi-$[$i+1].${PLM_NAME} \
         ${PROCESSED_DIR}/epi-${i}.${PLM_NAME} \
          &> $SCRIPT_LOG_DIR/epi-${i}.build.log && \
        echo "Starting Episode $[$i+1] training..." &&\
        source ./train.sh $[$i+1] ${CUDA_LIST[$i]} $PLM_DIR $MODEL_DIR/epi-$[$i+1].${PLM_NAME} \
         $PROCESSED_DIR/epi-$[$i+1].$PLM_NAME/ \
         $LOG_DIR/epi-$[$i+1].$PLM_NAME/ \
         $SCRIPT_LOG_DIR/epi-$[$i+1].train.log \
         $[ $EPISODE_NUM-$i-1 ] \

    fi

    if [ $i -ne $EPISODE_NUM ]; then # We only need to prepare for next episode if it exists
        # Copy the 20000 checkpoint to be used for next episode
        mkdir -p $MODEL_DIR/epi-$[$i+1].${PLM_NAME}-cp20000
        cp -v $MODEL_DIR/epi-$[$i+1].${PLM_NAME}/checkpoint-20000/* $MODEL_DIR/epi-$[$i+1].${PLM_NAME}-cp20000/
        # Copy tokenizer config from original model, to make copied model complete
        cp -v ${PLM_DIR}/special_tokens_map.json $MODEL_DIR/epi-$[$i+1].${PLM_NAME}-cp20000/
        cp -v ${PLM_DIR}/vocab.txt $MODEL_DIR/epi-$[$i+1].${PLM_NAME}-cp20000/
        cp -v ${PLM_DIR}/tokenizer_config.json $MODEL_DIR/epi-$[$i+1].${PLM_NAME}-cp20000/     
    fi
done

echo "Starting evaluation on final model..."
source ./evaluation.sh ${CUDA_LIST[$[$EPISODE_NUM-1]]} \
$MODEL_DIR/epi-${EPISODE_NUM}.${PLM_NAME} \
$EMBEDDING_DIR/epi-${EPISODE_NUM}.${PLM_NAME}-eval \
$RESULT_DIR/epi-${EPISODE_NUM}.${PLM_NAME}-eval \
 &> $SCRIPT_LOG_DIR/epi-${EPISODE_NUM}.eval.log
