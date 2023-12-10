mkdir -p $PLM_DIR
cd $PLM_DIR

wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/pytorch_model.bin
wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/tokenizer_config.json
wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/vocab.txt
wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/special_tokens_map.json
wget -c https://huggingface.co/Luyu/co-condenser-marco/resolve/main/config.json
