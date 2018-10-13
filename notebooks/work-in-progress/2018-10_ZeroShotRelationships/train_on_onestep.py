#from pytorch-openai-transformer-lm.model_pytorch import TransformerModel

import os, sys
sys.path.append('pytorch-openai-transformer-lm')

from model_pytorch import TransformerModel, load_openai_pretrained_model, DEFAULT_CONFIG

args = DEFAULT_CONFIG
model = TransformerModel(args)
load_openai_pretrained_model(model, path=os.path.join('.', 'orig', 'finetune-transformer-lm', 'model')+'/',
                                    path_names=os.path.join('.', 'orig', 'pytorch-openai-transformer-lm')+'/',
                            )

# So 