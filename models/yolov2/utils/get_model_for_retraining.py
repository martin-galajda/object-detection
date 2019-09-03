import tensorflow as tf

from data.openimages.constants import BoxableImagesConstants
from models.yolov2.constants import Constants as YOLOV2Constants
from keras.models import load_model, Model
from keras.layers import Conv2D, Lambda, Input
from models.yolov2.loss import lambda_loss
from models.yolov2.utils.load_anchors import load_anchors


def get_model_for_retraining():
    anchors = load_anchors()
    num_classes = BoxableImagesConstants.NUM_OF_CLASSES
    path_to_pretrained_yolov2 = YOLOV2Constants.PATH_TO_PRETRAINED_YOLO_V2_MODEL_FULLY_CONV

    pretrained_model = load_model(path_to_pretrained_yolov2)

    # topless_pretrained_yolo_v2.layers.pop()

    yolov2_top = Conv2D(
        filters = len(anchors) * (5 + num_classes),
        kernel_size=(1, 1),
        activation='linear',
        name='final_conv2d'
    )(pretrained_model.layers[-2].output)

    yolov2_body = Model(pretrained_model.layers[0].input, yolov2_top)

    # Create model input layers.
    # boxes_input = Input(shape=(None, 5), name='boxes_input')
    # detectors_mask_shape = (14, 14, 5, 1)
    # matching_boxes_shape = (14, 14, 5, 5)

    # detectors_mask_input = Input(shape=detectors_mask_shape, name="detectors_mask_input")
    # matching_boxes_input = Input(shape=matching_boxes_shape, name="matching_boxes_input")

    # Place model loss on CPU to reduce GPU memory usage.
    # with tf.device('/cpu:0'):
    #     # TODO: Replace Lambda with custom Keras layer for loss.
    #     yolo_loss_layer = Lambda(
    #         lambda_loss,
    #         # output_shape=(1, (None, 13, 13, None)),
    #         name='yolo_loss',
    #         arguments={
    #             'anchors': anchors,
    #             'num_classes': num_classes
    #         })
    #
    #     model_loss = yolo_loss_layer([
    #         yolov2_body.output,
    #         # boxes_input,
    #         # detectors_mask_input,
    #         # matching_boxes_input
    #     ])

    # yolov2_model_with_loss = Model([yolov2_body.input, boxes_input, detectors_mask_input, matching_boxes_input], model_loss)
    # yolov2_model_with_loss = Model(yolov2_body.input, yolov2_body.output)
    #
    # yolov2_model_with_loss.compile(optimizer='adam')

    # yolov2_body = Model(pretrained_model.layers[0].input, yolov2_top)

    # yolov2_body.compile(optimizer='adam')

    yolo_output_layer = Lambda(
        lambda yolo_output: yolo_output,
        # output_shape=(1, (None, 13, 13, None)),
        name='yolo_output')

    model_output = yolo_output_layer(
        [yolov2_body.output],
        # boxes_input,
        # detectors_mask_input,
        # matching_boxes_input
    )
    yolov2_output = Model([pretrained_model.layers[0].input], model_output)


    # return yolov2_body, yolov2_model_with_loss
    return yolov2_output
