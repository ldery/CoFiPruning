TASK=$1
SUFFIX=$2
PRUNING_TYPE=structured_heads+structured_mlp+hidden+layer
SPARSITY=0.95
DISTILL_LAYER_LOSS_ALPHA=0.9
LAYER_DISTILL_VERSION=4
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=$3
EX_CATE=$4
BERTTYPE=$5
DISTILL_CE_LOSS_ALPHA=$6
GPUID=$7
BASE_MODEL=bert-${BERTTYPE}-uncased

SUFFIX=${SUFFIX}_${DISTILLATION_PATH}_CEALPHA_${DISTILL_CE_LOSS_ALPHA}_BERTTYPE_${BERTTYPE}

echo $SUFFIX
echo $DISTILLATION_PATH
echo $BERTTYPE


bash scripts/run_CoFi.sh $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $SPARSITY_EPSILON $BASE_MODEL


# Run the evaluation script
./test.sh $TASK $SUFFIX $GPUID $BASE_MODEL $EX_CATE

# Gather the evaluation script results
FTPath=CoFI_runs/$TASK/$EX_CATE/$TASK\_$SUFFIX/best
writepath=$FTPath/results.log

for fs in $FTPath/*/all_log.txt; do
    arrIN=(${fs//// })
    config=${arrIN[-2]}
    result=$(grep 'Evaluating:' $fs | awk '{print $10}' | sort -n | tail -n 1)
    echo $config ' - ' $result >> $writepath
done


