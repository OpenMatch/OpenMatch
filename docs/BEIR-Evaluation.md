# BEIR Evaluation Guideline

## Data Download

Please download the latest BEIR dataset [[link]](https://github.com/beir-cellar/beir).

## Data Preprocess

For 18 datasets in BEIR, 16 of them can be directly loaded via OpenMatch. However, a small preprocessing is required before loading the`hotpotqa` and `fever` datasets, that is, to filter out the 'metadata' in the `queries.jsonl` file, otherwise an error will be reported when loading.


## Evaluation

```bash
## *************************************
export DATA_DIR=/data/private/sunsi/dataset/beir # dataset path
export train_job_name=ance-tele.cocodr-base # model path
## *************************************
## GPU setup
TOT_CUDA="1,2,3,4"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="1234"
## *************************************

## 17 BEIR Datasets
export OUTPUT_DIR=/data/private/sunsi/experiments/ance-tele
dataset_name_list=(trec-covid nfcorpus fiqa arguana webis-touche2020 quora scidocs scifact nq hotpotqa signal1m trec-news robust04 dbpedia-entity fever climate-fever bioasq)

# # CQADupStack dataset contains 12 small datasets.
# export DATA_DIR=/data/private/sunsi/dataset/beir/cqadupstack
# dataset_name_list=(android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress)

for dataset_name in ${dataset_name_list[@]}
do
    export infer_job_name=inference.${train_job_name}.${dataset_name}
    
    if [ ${dataset_name} == fiqa ] || [ ${dataset_name} == signal1m ]
    then
        export q_max_len=64
        export p_max_len=128
        echo ${infer_job_name}
        
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.beir_eval_pipeline \
        --data_dir ${DATA_DIR}/${dataset_name} \
        --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
        --output_dir ${OUTPUT_DIR}/${infer_job_name} \
        --query_template "<text>" \
        --doc_template "<text>" \
        --q_max_len ${q_max_len} \
        --p_max_len ${p_max_len}  \
        --per_device_eval_batch_size 4096  \
        --dataloader_num_workers 1 \
        --fp16 \
        --use_gpu \
        --overwrite_output_dir \
        --use_split_search \
        --max_inmem_docs 5000000 \
    
    
    elif [ ${dataset_name} == arguana ]
    then
        export q_max_len=128
        export p_max_len=128
        echo ${infer_job_name}
        
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.beir_eval_pipeline \
        --data_dir ${DATA_DIR}/${dataset_name} \
        --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
        --output_dir ${OUTPUT_DIR}/${infer_job_name} \
        --query_template "<text>" \
        --doc_template "<title> <text>" \
        --q_max_len ${q_max_len} \
        --p_max_len ${p_max_len}  \
        --per_device_eval_batch_size 4096  \
        --dataloader_num_workers 1 \
        --fp16 \
        --use_gpu \
        --overwrite_output_dir \
        --remove_identical \
        --use_split_search \
        --max_inmem_docs 5000000 \
    
    elif [ ${dataset_name} == quora ]
    then
        export q_max_len=64
        export p_max_len=128
        echo ${infer_job_name}
        
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.beir_eval_pipeline \
        --data_dir ${DATA_DIR}/${dataset_name} \
        --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
        --output_dir ${OUTPUT_DIR}/${infer_job_name} \
        --query_template "<text>" \
        --doc_template "<text>" \
        --q_max_len ${q_max_len} \
        --p_max_len ${p_max_len}  \
        --per_device_eval_batch_size 4096  \
        --dataloader_num_workers 1 \
        --fp16 \
        --use_gpu \
        --overwrite_output_dir \
        --remove_identical \
        --use_split_search \
        --max_inmem_docs 5000000 \
    
    
    elif [ ${dataset_name} == scifact ] || [ ${dataset_name} == trec-news ]
    then
        export q_max_len=64
        export p_max_len=256
        echo ${infer_job_name}
        
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.beir_eval_pipeline \
        --data_dir ${DATA_DIR}/${dataset_name} \
        --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
        --output_dir ${OUTPUT_DIR}/${infer_job_name} \
        --query_template "<text>" \
        --doc_template "<title> <text>" \
        --q_max_len ${q_max_len} \
        --p_max_len ${p_max_len}  \
        --per_device_eval_batch_size 512  \
        --dataloader_num_workers 1 \
        --fp16 \
        --use_gpu \
        --overwrite_output_dir \
        --use_split_search \
        --max_inmem_docs 5000000 \
        
        
    
    elif [ ${dataset_name} == robust04 ]
    then
        export q_max_len=64
        export p_max_len=256
        echo ${infer_job_name}
        
        
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.beir_eval_pipeline \
        --data_dir ${DATA_DIR}/${dataset_name} \
        --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
        --output_dir ${OUTPUT_DIR}/${infer_job_name} \
        --query_template "<text>" \
        --doc_template "<text>" \
        --q_max_len ${q_max_len} \
        --p_max_len ${p_max_len}  \
        --per_device_eval_batch_size 512  \
        --dataloader_num_workers 1 \
        --fp16 \
        --use_gpu \
        --overwrite_output_dir \
        --use_split_search \
        --max_inmem_docs 5000000 \
    
    
    else
        export q_max_len=64
        export p_max_len=128
        echo ${infer_job_name}
        
        CUDA_VISIBLE_DEVICES=${TOT_CUDA} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${CUDA_NUM} --master_port=${PORT} -m openmatch.driver.beir_eval_pipeline \
        --data_dir ${DATA_DIR}/${dataset_name} \
        --model_name_or_path ${OUTPUT_DIR}/${train_job_name} \
        --output_dir ${OUTPUT_DIR}/${infer_job_name} \
        --query_template "<text>" \
        --doc_template "<title> <text>" \
        --q_max_len ${q_max_len} \
        --p_max_len ${p_max_len}  \
        --per_device_eval_batch_size 4096  \
        --dataloader_num_workers 1 \
        --fp16 \
        --use_gpu \
        --overwrite_output_dir \
        --use_split_search \
        --max_inmem_docs 5000000 \
        
    fi
done



```