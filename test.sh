basefldr=$1
gpuid=$2
modelbase=$3
dirname=$4

CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 1e-4 $modelbase  True  $dirname
CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 1e-4 $modelbase  False $dirname

CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 3e-5 $modelbase  True  $dirname
CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 3e-5 $modelbase  False $dirname

CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 1e-5 $modelbase  True  $dirname
CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 1e-5 $modelbase  False $dirname

