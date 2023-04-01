TASK=$1
SUFFIX=$2
PRUNING_TYPE=structured_heads+structured_mlp+hidden+layer
SPARSITY=0.95
DISTILL_LAYER_LOSS_ALPHA=0.9
LAYER_DISTILL_VERSION=4
DISTILL_CE_LOSS_ALPHA=0.1
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=$3

FITNESS_STR=$4
BSZ=$5
LOCAL_THRESHOLDING=$6
PERROUND=$7
BERTTYPE=bert-$8-uncased
QUANTTHRESH=$9
GPUID=${10}
EX_CATE=${11}
SEED=${12}


SUFFIX=${SUFFIX}_${DISTILLATION_PATH}_${FITNESS_STR}_LOCALTHESH_${LOCAL_THRESHOLDING}_perrounds_${PERROUND}_BERT_$7_QUANTHRESH_$8_BSZ_$BSZ


bash scripts/run_CoFi.sh $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $SPARSITY_EPSILON $FITNESS_STR $BSZ $LOCAL_THRESHOLDING $PERROUND $BERTTYPE $QUANTTHRESH $SEED


# Run the evaluation script
./test.sh $TASK $SUFFIX $GPUID $BERTTYPE $EX_CATE $SEED

# Gather the evaluation script results
FTPath=CoFI_runs/$TASK/$EX_CATE/$TASK\_$SUFFIX/${SEED}
writepath=$FTPath/results.log

for fs in $FTPath/*/all_log.txt; do
    arrIN=(${fs//// })
    config=${arrIN[-2]}
    result=$(grep 'Evaluating:' $fs | awk '{print $10}' | sort -n | tail -n 1)
    echo $config ' - ' $result >> $writepath
done


