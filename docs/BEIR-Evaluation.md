# BEIR Evaluation Guideline

## Data Download

Please download the latest BEIR dataset [[link]](https://github.com/beir-cellar/beir).

## Data Preprocess

For 18 datasets in BEIR, 15 of them can be directly loaded via OpenMatch. However, a small preprocessing is required before loading the`hotpotqa`, `fever`, `robust04` datasets:

* `hotpotqa` and `fever` need to filter out the 'metadata' in the `queries.jsonl` file, otherwise an error will be reported when loading.
* `robust04` requires the following filtering operations in the `corpus.jsonl` file:
    ```
    ## robust04
    text = re.sub(r"[^A-Za-z0-9=(),!?\'\`]", " ", data['text'])
    text = " ".join(text.split())
    ```

## Evaluation

Here we use the `trec-covid` dataset as an example to describe the evaluation method. If you want to test all 18 datasets at once, you can run the shell script `OpenMatch/scripts/BEIR/eval_beir.sh`

```bash
## *************************************
## Model
export OUTPUT_DIR=/data/private/experiments # Path to store evaluation results.
export train_job_name=cocodr-base-msmarco # Folder of model files (Placed under OUTPUT_DIR by default).
## Dataset
export DATA_DIR=/data/private/dataset/beir # Path to store dataset files.
export dataset_name=trec-covid # Folder of dataset files (Placed under DATA_DIR by default).
## *************************************
## GPU setup
TOT_CUDA="1,2,3,4"
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="1234"
## *************************************
export q_max_len=64
export p_max_len=128
export infer_job_name=inference.${train_job_name}.${dataset_name}
## *************************************

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
```
