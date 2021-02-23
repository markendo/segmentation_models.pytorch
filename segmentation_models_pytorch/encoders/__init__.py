import functools
import torch.utils.model_zoo as model_zoo

from .resnet import resnet_encoders
from .dpn import dpn_encoders
from .vgg import vgg_encoders
from .senet import senet_encoders
from .densenet import densenet_encoders
from .inceptionresnetv2 import inceptionresnetv2_encoders
from .inceptionv4 import inceptionv4_encoders
from .efficientnet import efficient_net_encoders
from .mobilenet import mobilenet_encoders
from .xception import xception_encoders
from .timm_efficientnet import timm_efficientnet_encoders

from ._preprocessing import preprocess_input

import torch

encoders = {}
encoders.update(resnet_encoders)
encoders.update(dpn_encoders)
encoders.update(vgg_encoders)
encoders.update(senet_encoders)
encoders.update(densenet_encoders)
encoders.update(inceptionresnetv2_encoders)
encoders.update(inceptionv4_encoders)
encoders.update(efficient_net_encoders)
encoders.update(mobilenet_encoders)
encoders.update(xception_encoders)
encoders.update(timm_efficientnet_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None):
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)

    if weights is not None:
        if '/deep/group/aihc-bootcamp-spring2020/cxr_fewer_samples/experiments/ntruongv/' in weights: #moco weights
            state_dict = torch.load(weights)
            state_dict = state_dict['state_dict']
            for key in list(state_dict.keys()):
                if 'encoder_q' in key: # discard encoder_k, only use query not key
                    new_key = key.replace('module.encoder_q.', '')
                    state_dict[new_key] = state_dict[key]
                del state_dict[key]
            state_dict['fc.bias'] = 0
            state_dict['fc.weight'] = 0
            del state_dict['fc.0.weight']
            del state_dict['fc.0.bias']
            del state_dict['fc.2.weight']
            del state_dict['fc.2.bias']
        elif weights not in ['imagenet', 'ssl', 'swsl', 'instagram', 'advprop', 'noisy-student']: #chexpert weights
            state_dict = torch.load(weights)
            state_dict = state_dict['state_dict']
            for key in list(state_dict.keys()):
                new_key = key.replace('model.model.', '')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        else:
            settings = encoders[name]["pretrained_settings"][weights]
            state_dict = model_zoo.load_url(settings["url"])
        encoder.load_state_dict(state_dict)
    encoder.set_in_channels(in_channels)

    return encoder


def get_encoder_names():
    return list(encoders.keys())


def get_preprocessing_params(encoder_name, pretrained="imagenet"):
    if pretrained in ['imagenet', 'ssl', 'swsl', 'instagram', 'advprop', 'noisy-student']: # standard weights
        settings = encoders[encoder_name]["pretrained_settings"]
        if pretrained not in settings.keys():
            raise ValueError("Avaliable pretrained options {}".format(settings.keys()))
        formatted_settings = {}
        formatted_settings["input_space"] = settings[pretrained].get("input_space")
        formatted_settings["input_range"] = settings[pretrained].get("input_range")
        formatted_settings["mean"] = settings[pretrained].get("mean")
        formatted_settings["std"] = settings[pretrained].get("std")
        return formatted_settings
    else: # use CheXpert settings
        formatted_settings = {}
        formatted_settings["input_space"] = 'RGB'
        formatted_settings["input_range"] = [0, 1]
        formatted_settings["mean"] = [.5020, .5020, .5020]
        formatted_settings["std"] = [.085585, .085585, .085585]
        return formatted_settings


def get_preprocessing_fn(encoder_name, pretrained="imagenet"):
    params = get_preprocessing_params(encoder_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
