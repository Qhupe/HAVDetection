import cv2
import numpy as np
import matplotlib as mpl


def letterbox_image(image, size):

    ih, iw, ic = image.shape

    h, w = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    new_image = np.zeros((h, w, ic), dtype='uint8') + 128

    top, left = (h - nh) // 2, (w - nw) // 2

    new_image[top:top + nh, left:left + nw, :] = cv2.resize(image, (nw, nh))

    return new_image


def draw_detection(
        img,
        boxes,
        class_names,
        font=cv2.FONT_HERSHEY_DUPLEX,
        font_scale=0.5,
        box_thickness=2,
        border=5,
        text_color=(255, 255, 255),
        text_weight=1
):

    num_classes = len(class_names)
    colors = [mpl.colors.hsv_to_rgb((i / num_classes, 1, 1)) * 255 for i in range(num_classes)]

    for box in boxes:
        x1, y1, x2, y2 = box[:4].astype(int)
        score = box[-2]
        label = int(box[-1])

        clr = colors[label]

        img = cv2.rectangle(img, (x1, y1), (x2, y2), clr, box_thickness)

        text = f'{class_names[label]} ({score * 100:.0f}%)'

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)

        tb_x1 = x1 - box_thickness // 2
        tb_y1 = y1 - box_thickness // 2 - th - 2 * border
        tb_x2 = x1 + tw + 2 * border
        tb_y2 = y1

        img = cv2.rectangle(img, (tb_x1, tb_y1), (tb_x2, tb_y2), clr, -1)

        img = cv2.putText(img, text, (x1 + border, y1 - border), font, font_scale, text_color, text_weight, cv2.LINE_AA)

    return img