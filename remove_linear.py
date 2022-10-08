import torch
import sys


def remove_linear(finetune_model, pretrain_model, new_pretrain_model):
    finetune_checkpoint = torch.load(finetune_model, map_location="cpu")
    # print("finetune model has {} parts".format(len(finetune_checkpoint['model'].keys())))
    pretrain_checkpoint = torch.load(pretrain_model, map_location="cpu")
    # print("pretrain model has {} parts".format(len(pretrain_checkpoint['model'].keys())))
    for key in finetune_checkpoint['model'].keys():
        if key.startswith("w2v_encoder.w2v_model."):
            pretrain_key = key.replace("w2v_encoder.w2v_model.", "")
            assert pretrain_key in pretrain_checkpoint['model'].keys(), pretrain_key
            pretrain_checkpoint['model'][pretrain_key] = finetune_checkpoint['model'][key]
    torch.save(pretrain_checkpoint, new_pretrain_model)

if __name__ == "__main__":
    remove_linear(sys.argv[1], sys.argv[2], sys.argv[3])
