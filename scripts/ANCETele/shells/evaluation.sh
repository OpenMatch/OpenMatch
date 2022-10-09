#===========================
# Two variables are not passed as arguments; Please pass them as environment variables.
# COLLECTION_DIR: Path to the processed dataset files.
# OPENMATCH_SCRIPTS_DIR: Path to {OPENMATCH_INSTALL_DIRECTORY}/scripts.
# When called by openmatch-ANCETele.sh, these variables are set properly; Only set them when you directly call this module.
#===========================
#Comma-seperated index of CUDA device to use.
CUDA_TO_USE=$1
# Model used for evaluation.
CURRENT_MODEL_DIR=$2
# Path to save the embeddings created by the model.
EMBEDDING_SAVE_DIR=$3
# Path to the retrieved documents (TREC) and MRR@10 result.
RETRIEVE_SAVE_DIR=$4

CUDAs=(${CUDA_TO_USE//,/ })
CUDA_NUM=${#CUDAs[@]}
# Encode the corpus.
echo "Encoding corpus..."
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

# Then retrieve dev set document.
mkdir -p ${RETRIEVE_SAVE_DIR}
echo "Retrieving dev query documents using CUDA ${CUDA_TO_USE} ..."
CUDA_VISIBLE_DEVICES=$CUDA_TO_USE python -m openmatch.driver.retrieve  \
    --output_dir ${EMBEDDING_SAVE_DIR} \
    --model_name_or_path $CURRENT_MODEL_DIR  \
    --per_device_eval_batch_size 256  \
    --query_path $COLLECTION_DIR/dev.query.txt  \
    --query_template "<text>"  \
    --query_column_names id,text  \
    --q_max_len 32  \
    --fp16  \
    --trec_save_path ${RETRIEVE_SAVE_DIR}/dev.trec  \
    --dataloader_num_workers 1 \
    --use_gpu \

     # We only need 10 documents for evaluating MRR@10
# strip the trec file off any unwanted part leaving only "<query_id> <doc_id> <rank>"
awk '{if ( $4<=10 ) printf "%s %s %s\n",$1,$3,$4}' ${RETRIEVE_SAVE_DIR}/dev.trec | sed "s/ /\t/g" > ${RETRIEVE_SAVE_DIR}/dev.marco
# Finally, use official evaluation script to judge how well we've done
echo "Evaluating..."
python $OPENMATCH_SCRIPTS_DIR/ANCETele/ms_marco_eval.py ${COLLECTION_DIR}/qrels.dev.restructured.tsv ${RETRIEVE_SAVE_DIR}/dev.marco &> \
${RETRIEVE_SAVE_DIR}/dev_mrr.log
