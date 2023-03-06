TASK=RTE #CoLA #MNLI
DISTILL_CE_LOSS_ALPHA=$1
SPARSITY=$2
MODELBASE=$3
SUFFIX=FinetunedBert_${MODELBASE}_CEAlpha_${DISTILL_CE_LOSS_ALPHA}_sparse_${SPARSITY} # startwfinetuned_sparsity0.95 # 
EX_CATE=CoFi
PRUNING_TYPE=structured_heads+structured_mlp+hidden+layer
DISTILL_LAYER_LOSS_ALPHA=0.9
LAYER_DISTILL_VERSION=4
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=$4
# JeremiahZ/bert-base-uncased-rte
#bert-${MODELBASE}-uncased # JeremiahZ/bert-base-uncased-rte #
BASE_MODEL=bert-${MODELBASE}-uncased
# /mnt/lustre/sjtu/home/xc915/superb/CoFiPruning/teacher-model
echo $SUFFIX
echo $DISTILLATION_PATH
echo $BASE_MODEL

bash scripts/run_CoFi.sh $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $SPARSITY_EPSILON $BASE_MODEL
