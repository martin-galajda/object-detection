import numpy as np
def zeropad_boxes_by_max_num(batch_image_boxes, max_num_of_boxes_in_batch):
    zeropadded_results = []
    for batch_images in batch_image_boxes:
        shape_to_zeropad = np.array(batch_images).shape[1]
        len_difference_to_zeropad = max_num_of_boxes_in_batch - len(batch_images)

        if len_difference_to_zeropad > 0:
            new_zeropadded = np.zeros(len_difference_to_zeropad * shape_to_zeropad).reshape((len_difference_to_zeropad, shape_to_zeropad))
            if len(batch_images) > 0:
                new_results = np.vstack((batch_images, new_zeropadded))
            else:
                new_results = new_zeropadded
        else:
            new_results = batch_images

        zeropadded_results += [new_results]

    print(np.array(zeropadded_results).shape)
    return np.array(zeropadded_results)


def zeropad_boxes(batch_image_boxes):
    max_num_boxes = 0
    for image_boxes in batch_image_boxes:
        if len(image_boxes) > max_num_boxes:
            max_num_boxes = len(image_boxes)

    print(f'max_num_boxes = {max_num_boxes}')
    return zeropad_boxes_by_max_num(batch_image_boxes, max_num_boxes)
