TASK=RTE #CoLA #MNLI
SUFFIX=$1 #similar_sparsity #myapproach_10000 #randomized_ startwbertbase_sparsity0.95 # startwfinetuned_sparsity0.95 # 
EX_CATE=CoFi
PRUNING_TYPE=structured_heads+structured_mlp+hidden+layer
SPARSITY=0.95
DISTILL_LAYER_LOSS_ALPHA=0.9
DISTILL_CE_LOSS_ALPHA=0.1
LAYER_DISTILL_VERSION=4
SPARSITY_EPSILON=0.01
DISTILLATION_PATH=$2 #bert-base-uncased # JeremiahZ/bert-base-uncased-rte # 
FITNESS_STR=$3
BSZ=$4
# /mnt/lustre/sjtu/home/xc915/superb/CoFiPruning/teacher-model

echo $SUFFIX
echo $DISTILLATION_PATH

bash scripts/run_CoFi.sh $TASK $SUFFIX $EX_CATE $PRUNING_TYPE $SPARSITY $DISTILLATION_PATH $DISTILL_LAYER_LOSS_ALPHA $DISTILL_CE_LOSS_ALPHA $LAYER_DISTILL_VERSION $SPARSITY_EPSILON $FITNESS_STR $BSZ
