# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from siamg.models.backbone.mobile_v3 import mobilenetv3_small

BACKBONES = {
              'mobilenetv3_small': mobilenetv3_small,
            }

def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs) 
