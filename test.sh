basefldr=$1
gpuid=$2
modelbase=$3

CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 1e-4 $modelbase  True
# CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 1e-4 $modelbase  False

CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 3e-5 $modelbase  True
# CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 3e-5 $modelbase  False

CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 1e-5 $modelbase  True
# CUDA_VISIBLE_DEVICES=$gpuid ./runft.sh $basefldr 1e-5 $modelbase  False

