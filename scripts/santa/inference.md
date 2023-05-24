python build_test_code.py \
--input /data1/lixinze/dataset/go/test.jsonl \
--output /data1/lixinze/SANTA/codequery.jsonl \
--inference_data code \
--data_type query

python build_test_code.py \
--input /data1/lixinze/shop/esci_data/esci_data \
--output /data1/lixinze/SANTA/1.jsonl \
--inference_data product \
--data_type doc

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
    --output_dir /data1/lixinze/SANTA/product_embeddings/  \
    --model_name_or_path /data1/lixinze/shop/finetune-checkpoint/entity/t5-shop-5e-5_60/best_dev  \
    --per_device_eval_batch_size 256  \
    --query_path /data1/lixinze/SANTA/product_query.jsonl \
    --query_template "<query>"  \
    --q_max_len 20  \
    --trec_save_path /data1/lixinze/SANTA/product_inference.trec

python evaluate.py \
    --trec_save_path /data1/lixinze/SANTA/inference.trec \
    --data_type code 

python evaluate.py \
    --data_type product \
    --pickle_query_data /data1/lixinze/SANTA/product_embeddings/embeddings.query.rank.0 \
    --pickle_doc_data /data1/lixinze/SANTA/product_embeddings/embeddings.corpus.rank.0.0-181701 \
    --input /data1/lixinze/shop/esci_data/esci_data \
    --hypothesis_path_file /data1/lixinze/SANTA/product_trec/hypothesis.csv \
    --output_path /data1/lixinze/SANTA/product_trec

