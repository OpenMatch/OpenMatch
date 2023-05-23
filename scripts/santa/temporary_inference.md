python build_test_code.py \
--input /data1/lixinze/dataset/ruby/codebase.jsonl \
--output /data1/lixinze/SANTA/codebase.jsonl \
--data_type code


python -m openmatch.driver.build_index  \
    --output_dir /data1/lixinze/SANTA/embeddings/ \
    --model_name_or_path /data1/lixinze/dataset/ruby/finetune-checkpoint/hard_negative/ruby-SANTA128-0.5/SANTA-2e-5-128/best_dev  \
    --per_device_eval_batch_size 256  \
    --corpus_path /data1/lixinze/SANTA/codebase.jsonl  \
    --doc_template "<code>"  \
    --q_max_len 50  \
    --p_max_len 240  \
    --dataloader_num_workers 1

python -m openmatch.driver.retrieve  \
    --output_dir /data1/lixinze/SANTA/embeddings/  \
    --model_name_or_path /data1/lixinze/dataset/ruby/finetune-checkpoint/hard_negative/ruby-SANTA128-0.5/SANTA-2e-5-128/best_dev  \
    --per_device_eval_batch_size 256  \
    --query_path /data1/lixinze/SANTA/codequery.jsonl \
    --query_template "<nl>"  \
    --q_max_len 50  \
    --trec_save_path /data1/lixinze/SANTA/inference.trec

python evaluate.py \
    --trec_save_path /data1/lixinze/SANTA/inference.trec \
    --data_type code 

