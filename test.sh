task=$1
suffix=$2
gpuid=$3
modelbase=$4
dirname=$5
seed=$6


CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $task $suffix 1e-4 $modelbase  True  $dirname $seed
CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $task $suffix 1e-4 $modelbase  False $dirname $seed

CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $task $suffix 3e-5 $modelbase  True  $dirname $seed
CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $task $suffix 3e-5 $modelbase  False $dirname $seed

CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $task $suffix 1e-5 $modelbase  True  $dirname $seed
CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $task $suffix 1e-5 $modelbase  False $dirname $seed

