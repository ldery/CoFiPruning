TASK=$1 #CoLA #MNLI
SUFFIX=$2 #similar_sparsity #test_ #randomized_ #myapproach_ #  startwfinetuned_sparsity0.95 # startwbertbase_sparsity0.95 # 
proj_dir=CoFI_runs
LEARNING_RATE=$3
TEACHER_PATH=$4
RESET_LN=$5
EX_CATE=$6
SEED=$7


PRUNED_MODEL_PATH=$proj_dir/$TASK/$EX_CATE/${TASK}_${SUFFIX}/${SEED}


bash scripts/run_FT.sh $TASK $SUFFIX $LEARNING_RATE $PRUNED_MODEL_PATH $TEACHER_PATH $RESET_LN
