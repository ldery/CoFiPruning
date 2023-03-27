task=$1
basefldr=$2
gpuid=$3
modelbase=$4
dirname=$5

CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $task $basefldr 1e-4 $modelbase $dirname
CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $task $basefldr 3e-5 $modelbase $dirname
CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $task $basefldr 1e-5 $modelbase $dirname