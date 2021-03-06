import tensorflow as tf
import numpy as np
from models.faster_rcnn_inception_resnet_v2_oid_v4.constants import FasterRCNNPathConstants

OUTPUT_TENSOR_NAMES = [
    'num_detections',
    'detection_boxes',
    'detection_scores',
    'detection_classes'
]


def _construct_session_for_inference(
    *,
    path_to_frozen_inference_graph: str = FasterRCNNPathConstants.PATH_TO_FROZEN_TF_GRAPH,
    use_gpu: bool = True,
    allow_soft_placement: bool = True,
    log_device: bool = False,
    allow_gpu_memory_growth: bool = False,
    force_gpu_compatible: bool = True,
):
    graph = restore_inference_graph(
        path_to_frozen_inference_graph,
        use_gpu
    )
    with graph.as_default():
        config = tf.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device,
        )
        config.gpu_options.allow_growth = allow_gpu_memory_growth
        config.gpu_options.force_gpu_compatible = force_gpu_compatible

        session = tf.Session(config=config, graph=graph)

    return session


def restore_inference_graph(
    path_to_frozen_inference_graph: str = FasterRCNNPathConstants.PATH_TO_FROZEN_TF_GRAPH,
    use_gpu: bool = False,
):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_frozen_inference_graph, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)

            if use_gpu:
                with tf.device('/gpu:0'):
                    tf.import_graph_def(od_graph_def, name='')
            else:
                with tf.device('/cpu:0'):
                    tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def infer_objects_in_image(
    *,
    image: np.array,
    sess = tf.Session
):
    with sess.graph.as_default():
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}

        for key in OUTPUT_TENSOR_NAMES:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict