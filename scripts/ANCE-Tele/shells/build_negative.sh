#===========================
# Two variables are not passed as arguments; Please pass them as environment variables.
# COLLECTION_DIR: Path to the processed dataset files.
# OPENMATCH_SCRIPTS_DIR: Path to {OPENMATCH_INSTALL_DIRECTORY}/scripts.
# When called by openmatch-ANCETele.sh, these variables are set properly; Only set them when you directly call this module.
#===========================
#Episode of current building procedure. Used to decide if previous episode negatives needs to be merged.
EPISODE=$1
#Comma-seperated index of CUDA device to use.
CUDA_TO_USE=$2
# Model used to build the next step.
CURRENT_MODEL_DIR=$3
# Path to save the embeddings created by the model.
EMBEDDING_SAVE_DIR=$4
# Path to the retrieved documents (TREC)
RETRIEVE_SAVE_DIR=$5
# Path to save the train files.
# This path may need HUGE spaces(About 100 GiB), so make sure you have enough disk space for this.
TRAIN_SAVE_DIR=$6
# Path to train files used by last episode.
# This may be left blank when EPISODE=0.
LAST_TRAIN_SAVE_DIR=$7
# We first build the index for the corpus.
# This would create embeddings.corpus.rank.x file, a pickle of embedded corpus.
CUDAs=(${CUDA_TO_USE//,/ })
CUDA_NUM=${#CUDAs[@]}
echo "Encoding corpus..."
# Due to the index duplication bug, we fall back to use only one GPU for building
echo "Encoding on CUDA ${CUDAs[0]}"
CUDA_VISIBLE_DEVICES=${CUDAs[0]} python -m openmatch.driver.build_index  \
    --output_dir ${EMBEDDING_SAVE_DIR}  \
    --model_name_or_path ${CURRENT_MODEL_DIR}  \
    --per_device_eval_batch_size 1024  \
    --corpus_path ${COLLECTION_DIR}/corpus.tsv  \
    --doc_template "<title>[SEP]<text>"  \
    --doc_column_names id,title,text  \
    --q_max_len 32  \
    --p_max_len 128  \
    --fp16  \
    --dataloader_num_workers 1
# make Retrieve dir beforehand, since retrieve would not create this path,
mkdir -p ${RETRIEVE_SAVE_DIR}
echo "Retrieving train-positive negatives using CUDA ${CUDA_TO_USE} ..."
# Retriever would use all visible devices by default, so no need to distribute
# output_dir: path to embedded corpus files
# model_name_or_path: Path to model used, used for both tokenizing and embedding
# corpus_path: Containing training positives, which is tsv which passages correspond to qrels.train.tsv
# doc_template: Since dataset have no default template, this template has to be defined manually; however the [SEP] token is hard-coded here since retriever has no tokenizer-specfic conversion.

CUDA_VISIBLE_DEVICES=$CUDA_TO_USE python -m openmatch.driver.retrieve  \
    --output_dir ${EMBEDDING_SAVE_DIR} \
    --model_name_or_path $CURRENT_MODEL_DIR  \
    --per_device_eval_batch_size 256  \
    --corpus_path $COLLECTION_DIR/train.positive.txt  \
    --encode_query_as_passage \
    --doc_template "<title>[SEP]<text>"  \
    --doc_column_names id,title,text  \
    --p_max_len 128  \
    --retrieve_depth 200 \
    --fp16  \
    --use_gpu \
    --trec_save_path ${RETRIEVE_SAVE_DIR}/train.positive.trec  \
    --dataloader_num_workers 1

# Now we retrieve hard negatives as described in ds-marco-passage.md .
echo "Retrieving hard negatives using CUDA ${CUDA_TO_USE} ..."
CUDA_VISIBLE_DEVICES=$CUDA_TO_USE python -m openmatch.driver.retrieve  \
    --output_dir ${EMBEDDING_SAVE_DIR} \
    --model_name_or_path $CURRENT_MODEL_DIR  \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/train.query.txt  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --retrieve_depth 200 \
    --fp16  \
    --use_gpu \
    --trec_save_path ${RETRIEVE_SAVE_DIR}/train.trec  \
    --dataloader_num_workers 1

# We could first select&mix the negatives for training phase before fetching them, saving some encoding overhead.
# However, it's simpler to just build two training dataset before mixing them, so I'm not going the hard way for now.
echo "Building train datasets..."
# Build train dataset A from hard negatives
# Build train dataset B from train positive neighbors, PARALLELED
python $OPENMATCH_SCRIPTS_DIR/ANCETele/build_hn.py  \
    --tokenizer_name $CURRENT_MODEL_DIR  \
    --hn_file ${RETRIEVE_SAVE_DIR}/train.trec  \
    --qrels $COLLECTION_DIR/qrels.train.tsv  \
    --queries $COLLECTION_DIR/train.query.txt  \
    --collection $COLLECTION_DIR/corpus.tsv  \
    --save_to ${TRAIN_SAVE_DIR}/hard_neg  \
    --depth 200 \
    --n_sample 30 \
& \
python $OPENMATCH_SCRIPTS_DIR/ANCETele/build_hn.py  \
    --tokenizer_name $CURRENT_MODEL_DIR  \
    --hn_file ${RETRIEVE_SAVE_DIR}/train.positive.trec  \
    --qrels $COLLECTION_DIR/qrels.train.tsv  \
    --queries $COLLECTION_DIR/train.query.txt  \
    --collection $COLLECTION_DIR/corpus.tsv  \
    --save_to ${TRAIN_SAVE_DIR}/train_pos  \
    --depth 200 \
    --n_sample 30 \
    &
wait

# And zip the two datasets together
echo "Merging negatives..."
python $OPENMATCH_SCRIPTS_DIR/ANCETele/combine_negative.py \
--input_folder_1 ${TRAIN_SAVE_DIR}/train_pos \
--input_folder_2 ${TRAIN_SAVE_DIR}/hard_neg \
--output_folder ${TRAIN_SAVE_DIR}/cur_mixed \

# since training datasets could be easily rebuilt, we therefore discard all temporary files left behind to save disk space.
echo "Removing temporary datasets..."
rm -rv  ${TRAIN_SAVE_DIR}/hard_neg/
rm -rv  ${TRAIN_SAVE_DIR}/train_pos/

# If we're preprocessing at later EPISODEs, we also have to add previous negatives
if [ $EPISODE -gt 0 ]; then
echo "Merging episode $[$EPISODE -1 ] negatives..."
python $OPENMATCH_SCRIPTS_DIR/ANCETele/combine_negative.py \
--input_folder_1 ${TRAIN_SAVE_DIR}/cur_mixed \
--input_folder_2 ${LAST_TRAIN_SAVE_DIR} \
--output_folder ${TRAIN_SAVE_DIR}
else
mv -v ${TRAIN_SAVE_DIR}/cur_mixed/* ${TRAIN_SAVE_DIR}
fi
rm -rv  ${TRAIN_SAVE_DIR}/cur_mixed/
