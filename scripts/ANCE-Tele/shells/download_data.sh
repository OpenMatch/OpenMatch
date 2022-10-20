# Download datasets
mkdir -p $COLLECTION_DIR
cd $COLLECTION_DIR
if [ ! -f "para.txt" ]; then
    echo "Downloading datasets into $COLLECTION_DIR..."
    wget  https://rocketqa.bj.bcebos.com/corpus/marco.tar.gz
    tar -zxf marco.tar.gz 
    rm -rf marco.tar.gz
    mv -v ./marco/* ./
    wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv -O qrels.train.tsv
fi
# Merge the paragraph to title to create the corpus file.
echo "Merging corpus..."
join  -t "$(echo -en '\t')"  -e '' -a 1  -o 1.1 2.2 1.2  <(sort -k1,1 para.txt) <(sort -k1,1 para.title.txt) | sort -k1,1 -n > corpus.tsv

# And build train_positive file for ease of further use
echo "Processing train positives..."
python $OPENMATCH_SCRIPTS_DIR/ANCETele/build_train_positive.py --qrel_file $COLLECTION_DIR/qrels.train.tsv --corpus_file $COLLECTION_DIR/corpus.tsv --save_to $COLLECTION_DIR
# Restructure qrel of dev set to qrels form
echo "Processing qrel..."
awk '{printf "%s 0 %s 1\n",$1,$2}' $COLLECTION_DIR/qrels.dev.tsv | sed "s/ /\t/g" > $COLLECTION_DIR/qrels.dev.restructured.tsv