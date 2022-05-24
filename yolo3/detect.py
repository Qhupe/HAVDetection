import numpy as np
import tensorflow as tf

def detection(
    prediction,
    anchor_boxes,
    num_classes,
    image_shape,
    input_shape,
    max_boxes = 20,
    score_threshold=0.3,
    iou_threshold=0.45,
    classes_can_overlap=True,
):
    all_boxes = []

    for output, anchors in zip(prediction, anchor_boxes):

        batch_size = output.shape[0]
        grid_h, grid_w = output.shape[1:3]

        output = tf.reshape(output, [-1, grid_h, grid_w, len(anchors), num_classes + 5])

        anchors_tensor = tf.constant(anchors, dtype=output.dtype)

        image_shape_tensor = tf.cast(image_shape, output.dtype)
        grids_shape_tensor = tf.cast(output.shape[1:3], output.dtype)
        input_shape_tensor = tf.cast(input_shape, output.dtype)

        image_shape_tensor = tf.reshape(image_shape_tensor, [-1, 1, 1, 1, 2])
        grids_shape_tensor = tf.reshape(grids_shape_tensor, [-1, 1, 1, 1, 2])
        input_shape_tensor = tf.reshape(input_shape_tensor, [-1, 1, 1, 1, 2])

        sized_shape_tensor = tf.round(
            image_shape_tensor * tf.reshape(tf.reduce_min(input_shape_tensor / image_shape_tensor, axis=-1),
                                            [-1, 1, 1, 1, 1]))

        box_scaling = input_shape_tensor * image_shape_tensor / sized_shape_tensor / grids_shape_tensor

        box_offsets = (tf.expand_dims(tf.reduce_max(image_shape_tensor, axis=-1), axis=-1) - image_shape_tensor) / 2.

        grid_h, grid_w = output.shape[1:3]

        grid_i = tf.reshape(np.arange(grid_h), [-1, 1, 1, 1])
        grid_i = tf.tile(grid_i, [1, grid_w, 1, 1])

        grid_j = tf.reshape(np.arange(grid_w), [1, -1, 1, 1])
        grid_j = tf.tile(grid_j, [grid_h, 1, 1, 1])

        grid_ji = tf.concat([grid_j, grid_i], axis=-1)
        grid_ji = tf.cast(grid_ji, output.dtype)

        box_xy = output[..., 0:2]
        box_xy = tf.sigmoid(box_xy) + grid_ji

        box_wh = output[..., 2:4]
        box_wh = tf.exp(box_wh) * anchors_tensor

        box_xy = box_xy * box_scaling - box_offsets[..., ::-1]
        box_wh = box_wh * box_scaling

        box_x1_y1 = box_xy - box_wh / 2
        box_x2_y2 = box_xy + box_wh / 2

        box_x1_y1 = tf.maximum(0, box_x1_y1)
        box_x2_y2 = tf.minimum(box_x2_y2, image_shape_tensor[..., ::-1])

        if classes_can_overlap:
            classs_probs = tf.sigmoid(output[..., 4:5]) * tf.sigmoid(output[..., 5:])
        else:
            classs_probs = tf.sigmoid(output[..., 4:5]) * tf.nn.softmax(output[..., 5:])

        box_cl = tf.argmax(classs_probs, axis=-1)
        box_sc = tf.reduce_max(classs_probs, axis=-1)

        box_cl = tf.cast(box_cl, output.dtype)
        box_cl = tf.expand_dims(box_cl, axis=-1)
        box_sc = tf.expand_dims(box_sc, axis=-1)

        boxes = tf.reshape(tf.concat([box_x1_y1, box_x2_y2, box_sc, box_cl], axis=-1),
                           [batch_size, -1, 6])

        all_boxes.append(boxes)

    all_boxes = tf.concat(all_boxes, axis=1)

    all_final_boxes = []

    for _boxes_ in all_boxes:

        if classes_can_overlap:

            final_boxes = []

            for class_id in range(num_classes):
                class_boxes = _boxes_[_boxes_[..., -1] == class_id]

                selected_idc = tf.image.non_max_suppression(
                    class_boxes[..., :4],
                    class_boxes[..., -2],
                    max_output_size=max_boxes,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold
                )

                class_boxes = tf.gather(class_boxes, selected_idc)
                final_boxes.append(class_boxes)

            final_boxes = tf.concat(final_boxes, axis=0)

        else:

            selected_idc = tf.image.non_max_suppression(
                _boxes_[..., :4],
                _boxes_[..., -2],
                max_output_size=max_boxes,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold
            )

            final_boxes = tf.gather(_boxes_, selected_idc)

        all_final_boxes.append(final_boxes)

    return all_final_boxes