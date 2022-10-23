import numpy as np
from tensorflow.keras.layers import (Concatenate, Input, Lambda, MaxPooling2D,
                                     UpSampling2D, ZeroPadding2D)
from tensorflow.keras.models import Model

from nets.backbone import (DarknetConv2D, DarknetConv2D_BN_Leaky,
                           Multi_Concat_Block, darknet_body)
from nets.yolo_training import yolo_loss


def SPPCSPC(x, c2, n=1, shortcut=False, g=1, e=0.5, k=(13, 9, 5), weight_decay=5e-4, name=""):
    c_ = int(2 * c2 * e)  # hidden channels
    x1 = DarknetConv2D_BN_Leaky(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    
    y1 = Concatenate(axis=-1)([MaxPooling2D(pool_size=(m, m), strides=(1, 1), padding='same')(x1) for m in k] + [x1])
    y1 = DarknetConv2D_BN_Leaky(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv3')(y1)
    
    y2 = DarknetConv2D_BN_Leaky(c_, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    out = Concatenate(axis=-1)([y1, y2])
    out = DarknetConv2D_BN_Leaky(c2, (1, 1), weight_decay=weight_decay, name = name + '.cv4')(out)
    
    return out

#---------------------------------------------------#
#   Panet网络的构建，并且获得预测结果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes, weight_decay=5e-4):
    #-----------------------------------------------#
    #   定义了不同yolov7-tiny的参数
    #-----------------------------------------------#
    transition_channels = 16
    block_channels      = 16
    panet_channels      = 16
    e                   = 1
    n                   = 2
    ids                 = [-1, -2, -3, -4]
    #-----------------------------------------------#
    #   输入图片是640, 640, 3
    #-----------------------------------------------#

    inputs      = Input(input_shape)
    #---------------------------------------------------#   
    #   生成主干模型，获得三个有效特征层，他们的shape分别是：
    #   80, 80, 256
    #   40, 40, 512
    #   20, 20, 1024
    #---------------------------------------------------#
    feat1, feat2, feat3 = darknet_body(inputs, transition_channels, block_channels, n, weight_decay)

    # 20, 20, 1024 -> 20, 20, 512
    P5          = SPPCSPC(feat3, transition_channels * 16, weight_decay=weight_decay, name="sppcspc")
    P5_conv     = DarknetConv2D_BN_Leaky(transition_channels * 8, (1, 1), weight_decay=weight_decay, name="conv_for_P5")(P5)
    P5_upsample = UpSampling2D()(P5_conv)
    P4          = Concatenate(axis=-1)([DarknetConv2D_BN_Leaky(transition_channels * 8, (1, 1), weight_decay=weight_decay, name="conv_for_feat2")(feat2), P5_upsample])
    P4          = Multi_Concat_Block(P4, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids, weight_decay=weight_decay, name="conv3_for_upsample1")

    P4_conv     = DarknetConv2D_BN_Leaky(transition_channels * 4, (1, 1), weight_decay=weight_decay, name="conv_for_P4")(P4)
    P4_upsample = UpSampling2D()(P4_conv)
    P3          = Concatenate(axis=-1)([DarknetConv2D_BN_Leaky(transition_channels * 4, (1, 1), weight_decay=weight_decay, name="conv_for_feat1")(feat1), P4_upsample])
    P3          = Multi_Concat_Block(P3, panet_channels * 2, transition_channels * 4, e=e, n=n, ids=ids, weight_decay=weight_decay, name="conv3_for_upsample2")
        
    P3_downsample = ZeroPadding2D(((1, 1),(1, 1)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(transition_channels * 8, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'down_sample1')(P3_downsample)
    P4 = Concatenate(axis=-1)([P3_downsample, P4])
    P4 = Multi_Concat_Block(P4, panet_channels * 4, transition_channels * 8, e=e, n=n, ids=ids, weight_decay=weight_decay, name="conv3_for_downsample1")

    P4_downsample = ZeroPadding2D(((1, 1),(1, 1)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(transition_channels * 16, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = 'down_sample2')(P4_downsample)
    P5 = Concatenate(axis=-1)([P4_downsample, P5])
    P5 = Multi_Concat_Block(P5, panet_channels * 8, transition_channels * 16, e=e, n=n, ids=ids, weight_decay=weight_decay, name="conv3_for_downsample2")
    
    P3 = DarknetConv2D_BN_Leaky(transition_channels * 8, (3, 3), strides=(1, 1), weight_decay=weight_decay, name="rep_conv_1")(P3)
    P4 = DarknetConv2D_BN_Leaky(transition_channels * 16, (3, 3), strides=(1, 1), weight_decay=weight_decay, name="rep_conv_2")(P4)
    P5 = DarknetConv2D_BN_Leaky(transition_channels * 32, (3, 3), strides=(1, 1), weight_decay=weight_decay, name="rep_conv_3")(P5)

    # len(anchors_mask[2]) = 3
    # 5 + num_classes -> 4 + 1 + num_classes
    # 4是先验框的回归系数，1是sigmoid将值固定到0-1，num_classes用于判断先验框是什么类别的物体
    # bs, 20, 20, 3 * (4 + 1 + num_classes)
    out2 = DarknetConv2D(len(anchors_mask[2]) * (5 + num_classes), (1, 1), weight_decay=weight_decay, strides = (1, 1), name = 'yolo_head_P3')(P3)
    out1 = DarknetConv2D(len(anchors_mask[1]) * (5 + num_classes), (1, 1), weight_decay=weight_decay, strides = (1, 1), name = 'yolo_head_P4')(P4)
    out0 = DarknetConv2D(len(anchors_mask[0]) * (5 + num_classes), (1, 1), weight_decay=weight_decay, strides = (1, 1), name = 'yolo_head_P5')(P5)
    return Model(inputs, [out0, out1, out2])

def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), 2)) for l in range(len(anchors_mask))] + [Input(shape = [None, 5])]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {
            'input_shape'       : input_shape, 
            'anchors'           : anchors, 
            'anchors_mask'      : anchors_mask, 
            'num_classes'       : num_classes, 
            'label_smoothing'   : label_smoothing, 
            'balance'           : [0.4, 1.0, 4],
            'box_ratio'         : 0.05,
            'obj_ratio'         : 1 * (input_shape[0] * input_shape[1]) / (640 ** 2), 
            'cls_ratio'         : 0.5 * (num_classes / 80)
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model
