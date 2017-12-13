import os
import argparse


class FormatTrainSet:
    def __init__(self):
        return

    def format(self, image_dir, label_dir, txt_dir, image_type, image_format='jpg', name_only=False):
        image_files = os.listdir(label_dir)

        images = []
        annots = []

        for id_ in image_files:
            file_name = id_.split('.')
            file_ex = file_name[1]
            if file_ex != 'png' and file_ex != 'jpg':
                continue
            file_name = file_name[0]
            # image_name = file_name.rstrip("_L")
            # image_name = image_name.lstrip("a-")

            if name_only:
                images.append('{}.{}'.format(file_name, image_format))
                annots.append('{}.png'.format(file_name))
            else:
                images.append(os.path.join(image_dir, '{}.{}'.format(file_name, image_format)))
                annots.append(os.path.join(label_dir, '{}.png'.format(file_name)))

        images.sort()
        annots.sort()
        image_count = len(images)
        label_count = len(annots)

        train_txt = os.path.join(txt_dir, '{}.txt'.format(image_type))
        with open(train_txt, 'wb') as f:
            if image_count == label_count:
                for image, annot in zip(images, annots):
                    str = image + '\t' + annot + '\n'
                    f.write(str.encode("UTF-8"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=False)
    parser.add_argument('--image_format', type=str, required=False)
    parser.add_argument('--annot_dir', type=str, required=False)
    parser.add_argument('--text_dir', type=str, required=False)
    args = parser.parse_args()

    handle = FormatTrainSet()
    image_dir = args.image_dir
    annot_dir = args.annot_dir
    txt_dir = args.text_dir
    image_type = 'train'
    image_format = 'jpg'
    if args.image_format:
        image_format = args.image_format
    handle.format(
        image_dir=image_dir,
        label_dir=annot_dir,
        txt_dir=txt_dir,
        image_type=image_type,
        image_format=image_format,
        name_only=False
    )


