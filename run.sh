#!/bin/bash


export CUDA_VISIBLE_DEVICES=0


####################
# This script computes the layer-wise CCA similarity between the pre-trained and finetuned wav2vec 2.0 model on the aishell dev set.
# The files to be prepared before running this scripts:
# 1. The pre-trained wav2vec model "exp/wav2vec/wav2vec_small.pt". It is the official wav2vec 2.0 pre-trained model: https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt.
#    Our link: https://drive.google.com/file/d/1uJKEIpagoZzVnPKudbHm8_EbTo8miUvk/view?usp=sharing
# 2. and the fine-tuned wav2vec model "exp/wav2vec_finetune/checkpoint_best.pt". It is the wav2vec-ctc model finetuned on the 10h aishell training subset.
#    Our link: https://drive.google.com/file/d/1lWyeSSDDrgTjhjN5oVIJNFw1lYhsiUCD/view?usp=sharing
# 3. The tsv file for the aishell dev "data/aishell/dev.tsv".
#    Our link: https://drive.google.com/file/d/1mjXD-YggcLPyyQZXEq70taTfA4m4HDVS/view?usp=sharing
#####################


corpus_name=aishell
model_names="wav2vec wav2vec_finetune"
data_dir=data/${corpus_name}
data_split=dev

stage=0
stop_stage=2

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "Stage 0: Prepare checkpoints for feature extraction"
    if [ ! -f exp/wav2vec/wav2vec_small.pt ] || [ ! -f exp/wav2vec_finetune/checkpoint_best.pt ];then 
        echo "Please prepare the checkpoints."
    fi

    if [ -f exp/wav2vec/checkpoint.pt ];then 
        rm exp/wav2vec/checkpoint.pt || true;
    fi
    ln -s wav2vec_small.pt exp/wav2vec/checkpoint.pt

    if [ -f exp/wav2vec_finetune/checkpoint.pt ];then 
        rm exp/wav2vec_finetune/checkpoint.pt || true;
    fi
    python remove_linear.py exp/wav2vec_finetune/checkpoint_best.pt exp/wav2vec/checkpoint.pt exp/wav2vec_finetune/checkpoint.pt
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "Stage 1: Extract features."
    for model_name in ${model_names};do
        for layer in `seq 0 11`;do
            checkpoint=exp/${model_name}/checkpoint.pt
            save_dir=save/${model_name}/${corpus_name}
            mkdir -p ${save_dir} || true;
            echo "Extracting features from the ${layer}-th layer of the ${model_name} model"
            python extract_features.py --data-dir ${data_dir} --split ${data_split} --save-dir ${save_dir}/layer${layer} --checkpoint ${checkpoint} --layer ${layer}
        done
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "Stage 2: Compute CCA similarities."
    for layer in `seq 0 11`;do
        python pwcca.py save/wav2vec/aishell/layer${layer}/${data_split}.npy save/wav2vec_finetune/aishell/layer${layer}/${data_split}.npy
    done
fi