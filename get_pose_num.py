# Import TF and TF Hub libraries.
import tensorflow as tf
import tensorflow_hub as hub
import csv
from PIL import Image

good_arr = ['ex-1.png', 'ex-2.png','ex-3.png','ex-4.png','ex-5.png','ex-6.png','ex-7.png','ex-8.png','ex-9.png','ex-10.png']
bad_arr = ['bad-6.png','bad-7.png']
f = open('train-data.csv', 'w');
header = ['nose', 'nose', 'nose',
          'left eye', 'left eye', 'left eye',
          'right eye', 'right eye', 'right eye',
          'left ear', 'left ear', 'left ear',
          'right ear', 'right ear', 'right ear',
          'left shoulder', 'left shoulder', 'left shoulder',
          'right shoulder', 'right shoulder', 'right shoulder',
          'left elbow', 'left elbow', 'left elbow',
          'right elbow', 'right elbow', 'right elbow',
          'left wrist', 'left wrist', 'left wrist',
          'right wrist', 'right wrist', 'right wrist',
          'left hip', 'left hip', 'left hip',
          'right hip', 'right hip', 'right hip',
          'left knee', 'left knee', 'left knee',
          'right knee', 'right knee', 'right knee',
          'left ankle', 'left ankle', 'left ankle',
          'right ankle', 'right ankle', 'right ankle', 'decision']
writer = csv.writer(f)
writer.writerow(header)

def get_and_write_posture(image_arr, good_posture):
    global header
    for image_path in image_arr:

        png = Image.open(image_path)
        png.load()  # required for png.split()

        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel

        background.save('foo.jpg', 'JPEG', quality=80)

        image = tf.io.read_file('foo.jpg')
        image = tf.compat.v1.image.decode_jpeg(image)
        image = tf.expand_dims(image, axis=0)
        # Resize and pad the image to keep the aspect ratio and fit the expected size.
        image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

        # Download the model from TF Hub.
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        movenet = model.signatures['serving_default']

        # Run model inference.
        outputs = movenet(image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints = outputs['output_0']

        csv_values = []

        for arr in keypoints[0][0]:
            csv_values.append(tf.get_static_value(arr[0]))
            csv_values.append(tf.get_static_value(arr[1]))
            csv_values.append(tf.get_static_value(arr[2]))

        if (good_posture):
            csv_values.append('good');
        else:
            csv_values.append('bad');


        writer.writerow(csv_values)


get_and_write_posture(good_arr, True)
get_and_write_posture(bad_arr, False)







