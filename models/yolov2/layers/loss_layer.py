from keras import backend as K
from keras.layers import Layer

class YoloV2LossLayer(Layer):
    def __init__(self,
        *,
        anchors,
        coord_loss_weight = 5.,
        objectness_loss_weight = 1.,
        incorrect_objectness_loss_weight = 0.5
    ):
        self.anchors = anchors

        super(YoloV2LossLayer, self).__init__()

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(YoloV2LossLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        yolov2_feats, true_boxes, detectors_mask_input, matching_boxes_input = x

        # predicted centers (x, y)
        # shape (batch_size, grid_height, grid_width, 2)
        pred_center_xy = K.sigmoid(yolov2_feats[..., :2])

        # predicted (width, height)  -> (e ^ (p_w_feat) * anchor_width, e ^ (p_h_feat) * anchor_height)
        # shape (batch_size, grid_height, grid_width, 2) * (5,2) ---> shape (batch_size, grid_height, grid_width, 5)
        pred_wh = K.exp(yolov2_feats[..., 2: 4]) * self.anchors

        # shape (batch_size, grid_height, grid_width, 1)
        pred_obj_confidence = K.sigmoid(yolov2_feats[..., 4])

        # shape (batch_size, grid_height, grid_width, number_of_classes)
        pred_classes = yolov2_feats[..., 5:]

        pred_start_xy = pred_center_xy - 0.5 * pred_wh
        pred_end_xy = pred_center_xy + 0.5 * pred_wh

        true_center_xy = true_boxes[..., :2]
        true_wh = true_boxes[..., 2:4]

        true_start_xy = true_center_xy - 0.5 * true_wh
        true_end_xy = true_center_xy + 0.5 * true_wh

        intersection_begin = K.maximum(pred_start_xy, true_start_xy)
        intersection_end = K.minimum(pred_end_xy, true_end_xy)
        intersection_diff = intersection_end - intersection_begin
        intersection_area = K.maximum(intersection_diff[..., 0] * intersection_diff[..., 1], 0)

        union_area = pred_wh[..., 0] * pred_wh[..., 1] + true_wh[..., 0] * true_wh[..., 1] - intersection_area

        # shape (batch_size, grid_height, grid_width, 1) ?
        intersection_over_union = intersection_area / union_area
        detected_object_mask 


        return []

    def compute_output_shape(self, input_shape):
        return (4,)
