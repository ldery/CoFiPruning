TASK=RTE #CoLA #MNLI
SUFFIX=$1 #similar_sparsity #myapproach_10000 #randomized_ startwbertbase_sparsity0.95 # startwfinetuned_sparsity0.95 # 
PRUNING_TYPE=structured_heads+structured_mlp+hidden+layer
SPARSITY=0.95
DISTILL_LAYER_LOSS_ALPHA=0.9
DISTILL_CE_LOSS_ALPHA=0.1
LAYER_DISTILL_VERSION=4
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=$2 #bert-base-uncased # JeremiahZ/bert-base-uncased-rte # 
FITNESS_STR=$3
BSZ=$4
LOCAL_THRESHOLDING=$5
PERROUND=$6
BERTTYPE=bert-$7-uncased
QUANTTHRESH=$8
GPUID=$9
EX_CATE=${10}
SUFFIX=${SUFFIX}_${DISTILLATION_PATH}_${FITNESS_STR}_LOCALTHESH_${LOCAL_THRESHOLDING}_perrounds_${PERROUND}_BERT_$7_QUANTHRESH_$8_BSZ_$BSZ
# DISTILLATION_PATH=JeremiahZ/bert-base-uncased-rte
# /mnt/lustre/sjtu/home/xc915/superb/CoFiPruning/teacher-model

echo $SUFFIX
echo $DISTILLATION_PATH
echo $BERTTYPE

bash scripts/run_CoFi.sh $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $SPARSITY_EPSILON $FITNESS_STR $BSZ $LOCAL_THRESHOLDING $PERROUND $BERTTYPE $QUANTTHRESH

# Run the evaluation script
./test.sh $SUFFIX $GPUID $BERTTYPE $EX_CATE

# Gather the evaluation script results
FTPath=CoFI_runs/$TASK/$EX_CATE/$TASK\_$SUFFIX
writepath=$FTPath/results.log

for fs in $FTPath/*/all_log.txt; do
    arrIN=(${fs//// })
    config=${arrIN[-2]}
    result=$(grep 'Evaluating:' $fs | awk '{print $10}' | sort -n | tail -n 1)
    echo $config ' - ' $result >> $writepath
done


