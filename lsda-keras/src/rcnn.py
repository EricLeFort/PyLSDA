from math import ceil

import pickle
import numpy as np
from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPool2D, Dense, Dropout, Activation, Concatenate, 
                          Lambda, ZeroPadding2D, Flatten)
from skimage.transform import resize

from LRN2D import LRN2D

def extract_regions(img, boxes, model):
    """
    Extract image regions and preprocess them into mean-subtracted batches for the model.

    Args:
        img (ndarray): The image
        boxes (list of Box): The (x1, y1, x2, y2) boxes
        model ():
    Returns:
        A 2-tuple containing:
            batches: The resulting 5-D batch of region images in the format:
                (batch x box x channel x width x height)
            batch_padding: The padding for the batch
    """
    num_batches = ceil(len(boxes) / model.cnn.batch_size)
    batch_padding = model.cnn.batch_size - len(boxes) % model.cnn.batch_size

    # Allocate memory up-front
    target_size = model.cnn.image_mean.shape[0]
    batches = np.ndarray(
        (num_batches, model.cnn.batch_size, 3, crop_size, crop_size),
        dtype="float32"
    )

    # Create the batches
    for i, _ in enumerate(batches):
        start = (i-1)*model.cnn.batch_size + 1
        end = min(len(boxes), start+model.cnn.batch_size-1)
        for j, box in enumerate(boxes[start:end]):
            batches[i, j, :, :, :] = img_crop(img, box, model)

    return batches, batch_padding

def features(img, boxes, model):
    """
    Compute CNN features on a set of boxes

    Args:
        img (ndarray): The image
        boxes (list of tuple): The (x1, y1, x2, y2) boxes
        model (Model): The model to use
    Returns:
        The resulting features
    """
    print("Extract regions... ", end="")
    batches, batch_padding = extract_regions(img, boxes, model)
    batch_size = model.cnn.batch_size
    print("Done.")

    # Compute features for each batch of region images
    print("Computing features... ", end="")
    feat_dim = -1
    feat = []
    idx = 1
    for i, batch in enumerate(batches):
        x = model.forward(batch)[0]

        # First batch, init values
        if i == 0:
            feat_dim = x.shape[0] / batch_size
            feat = np.zeros(len(boxes), feat_dim, dtype="float32")

        x = x.reshape((feat_dim, batch_size))

        # Last batch, trim x to size
        if i == len(batches) - 1:
            if batch_padding > 0:
                x = x[:, :x.shape[1] - batch_padding]

        feat[idx:idx+x.shape[1], :] = x
        idx += batch_size

    print("Done")
    return feat

def img_crop(img, box, model):
    """

    Args:
        img (ndarray): The image
        box (Box): The (x1, y1, x2, y2) box
        model ():
    """
    #mode = model.detectors.crop_mode
    #img_mean = model.cnn.image_mean
    #padding = model.detectors.crop_padding
    size = model.cnn.image_mean.shape[0]

    # Default values if padding is 0
    pad_x, pad_y = 0, 0
    crop_width, crop_height = size, size

    # Determine padding if necessary
    if padding > 0 or mode.detectors.crop_mode == "square":
        scale = size / (size - 2*model.detectors.crop_padding)
        half_height, half_width = (box[3] - box[1] + 1) / 2, (box[2] - box[0] + 1) / 2
        center = round((box[0] + half_width, box[1] + half_height))

        # Square off using the large dimension
        if mode.detectors.crop_mode == "square":
            half_height = max(half_height, half_width)
            half_width = half_height

        # Re-position according to scale
        half_height *= scale
        half_width *= scale
        box = (
            center - half_width,
            center - half_height,
            center + half_width,
            center + half_height
        )

        org_width, org_height = box[2] - box[0], box[3] - box[1]
        pad_x, pad_y = max(0, 1-box[0]), max(0, 1-box[1])

        # Clipped box
        box = (
            max(1, 1-box[0]),
            max(1, 1-box[1]),
            min(img.shape[1], box[2]),
            min(img.shape[0], box[3])
        )
        clipped_width, clipped_height = box[2] - box[0], box[3] - box[1]

        # Re-scale
        scale_x, scale_y = size / org_width, size / org_height
        crop_width, crop_height = round(clipped_width * scale_x), round(clipped_height * scale_y)
        pad_x, pad_y = round(pad_x * scale_x), round(pad_y * scale_y)

        # Determine final crop dimensions
        if pad_x + crop_width > size:
            crop_width = size - pad_x
        if pad_y + crop_height > size:
            crop_height = size - pad_y

    # Grab the region of the image we're interested in
    window = img[:, box[0]:box[2], box[1]:box[3]]

    # resize image, order=1 means "bilinear", the goal is to match what the model is doing
    tmp = resize(window, (crop_height, crop_width), order=1, anti_aliasing=False)

    # Subtract image mean
    if model.cnn.image_mean:
        tmp -= model.cnn.image_mean[:, pad_x:pad_x+crop_width, pad_y:pad_y+crop_height]

    window = np.zeros((3, size, size), dtype="float32")
    window[:, pad_x:pad_x+crop_width, pad_y:pad_y+crop_height] = tmp
    return window

def lX_to_fcX(feat, precomp_layer, layer, model):
    """
    On-the-fly conversion of some layer (5 or higher) features to a final fully-connected layer
    using the weights and biases stored in model.layers

    Args:
        feat ():
        precomp_layer (int):
        layer (int):
        model (Model): The model to use
    """
    for i in range(precomp_layer+1:layer):
        feat = feat*model.layers[i].weights[0] + model.layers[i].weights[1]
        if i < len(model.layers):
            feat = max(0, feat)

def pool5_to_fcX(feat, layer, model):
    """
    On-the-fly conversion of pool5 features to fc6 or fc7 using the weights and biases stored in
    model.layers

    Args:
        feat ():
        layer ():
        model (Model): The model to use
    """
    for i in range(5, layer):
        feat = max(0, feat*model.layers[i].weights[0] + model.layers[i].weights[1])

def scale_features(feat, feat_norm_mean, target_norm=20):
    """
    Scales the features according to the target norm

    Args:
        feat ():
        feat_norm_mean(): 
        target_norm (number): The target norm value
    Returns:
        The scaled features
    """
    return feat * (target_norm / feat_norm_mean)

def _get_conv_weights_and_biases(layer_data):
    """
    Reformats the weights in the data to the expected order
    
    Args:
        layer_data (list of ndarray): The weights and biases for a layer
    Returns:
        The altered weights and biases
    """
    return np.transpose(layer_data[0], (3, 2, 1, 0)), layer_data[1]

def _get_fc_weights_and_biases(layer_data):
    """
    Reformats the weights in the data to the expected order
    
    Args:
        layer_data (list of ndarray): The weights and biases for a layer
    Returns:
        The altered weights and biases
    """
    return np.transpose(layer_data[0], (1, 0)), layer_data[1]

def _load_weights(model, filepath):
    """
    Loads the saved model data (translated from Caffe) into this model
    
    Args:
        model (Model): The model to use
        filepath (str): The path to the file containing the model weights
    Returns:
        None
    """
    with open(filepath, "rb") as f:
        data = pickle.load(f)
        
    model.get_layer("conv1").set_weights(_get_conv_weights_and_biases(
        [layer for layer in data if layer["name"] == "conv1"][0]["weights"]
    ))
    
    weights, biases = _get_conv_weights_and_biases(
        [layer for layer in data if layer["name"] == "conv2"][0]["weights"]
    )
    model.get_layer("conv2_1").set_weights([weights[:, :, :, :128], biases[:128]])
    model.get_layer("conv2_2").set_weights([weights[:, :, :, 128:], biases[128:]])
    
    model.get_layer("conv3").set_weights(_get_conv_weights_and_biases(
        [layer for layer in data if layer["name"] == "conv3"][0]["weights"]
    ))
    
    weights, biases = _get_conv_weights_and_biases(
        [layer for layer in data if layer["name"] == "conv4"][0]["weights"]
    )
    model.get_layer("conv4_1").set_weights([weights[:, :, :, :192], biases[:192]])
    model.get_layer("conv4_2").set_weights([weights[:, :, :, 192:], biases[192:]])
    
    weights, biases = _get_conv_weights_and_biases(
        [layer for layer in data if layer["name"] == "conv5"][0]["weights"]
    )
    model.get_layer("conv5_1").set_weights([weights[:, :, :, :128], biases[:128]])
    model.get_layer("conv5_2").set_weights([weights[:, :, :, 128:], biases[128:]])
    
    model.get_layer("fc6").set_weights(_get_fc_weights_and_biases(
        [layer for layer in data if layer["name"] == "fc6"][0]["weights"]
    ))
    
    model.get_layer("fc7").set_weights(_get_fc_weights_and_biases(
        [layer  for layer in data if layer["name"] == "fc7"][0]["weights"]
    ))
    
    model.get_layer("fc8-7k").set_weights(_get_fc_weights_and_biases(
        [layer for layer in data if layer["name"] == "fc8-7k"][0]["weights"]
    ))

def _create_model(img_size=227):
    """
    Creates the architecture for the model
    
    Args:
        img_size (int): The size of the input image
    Returns:
        The created model
    """
    input_ = Input(shape=(img_size, img_size, 3), name='input')
    
    conv_1 = Conv2D(96, (11, 11), strides=4, use_bias=True, input_shape=(img_size, img_size, 3), activation='relu', name="conv1")(input_)
    conv_1 = MaxPool2D(pool_size=3, strides=2, name="pool1")(conv_1)
    conv_1 = LRN2D(name="norm1")(conv_1)
    conv_1 = ZeroPadding2D(padding=2)(conv_1)
    
    conv_1_a = Lambda(lambda x: x[:, :, :, :48])(conv_1)
    conv_1_b = Lambda(lambda x: x[:, :, :, 48:])(conv_1)
    conv_2_1 = Conv2D(128, (5, 5), use_bias=True, activation='relu', name="conv2_1")(conv_1_a)
    conv_2_2 = Conv2D(128, (5, 5), use_bias=True, activation='relu', name="conv2_2")(conv_1_b)
    conv_2 = Concatenate(name="conv2")([conv_2_1, conv_2_2])
    conv_2 = MaxPool2D(pool_size=3, strides=2, name="pool2")(conv_2)
    conv_2 = LRN2D(name="norm2")(conv_2)
    conv_2 = ZeroPadding2D(padding=1)(conv_2)
    
    conv_3 = Conv2D(384, (3, 3), use_bias=True, activation='relu', name="conv3")(conv_2)
    conv_3 = ZeroPadding2D(padding=1)(conv_3)
    
    conv_3_a = Lambda(lambda x: x[:, :, :, :192])(conv_3)
    conv_3_b = Lambda(lambda x: x[:, :, :, 192:])(conv_3)
    conv_4_1 = Conv2D(192, (3, 3), use_bias=True, activation='relu', name="conv4_1")(conv_3_a)
    conv_4_2 = Conv2D(192, (3, 3), use_bias=True, activation='relu', name="conv4_2")(conv_3_b)
    conv_4 = Concatenate(name="conv4")([conv_4_1, conv_4_2])
    conv_4 = ZeroPadding2D(padding=1)(conv_4)
    
    conv_4_a = Lambda(lambda x: x[:, :, :, :192])(conv_4)
    conv_4_b = Lambda(lambda x: x[:, :, :, 192:])(conv_4)
    conv_5_1 = Conv2D(128, (3, 3), use_bias=True, activation='relu', name="conv5_1")(conv_4_a)
    conv_5_2 = Conv2D(128, (3, 3), use_bias=True, activation='relu', name="conv5_2")(conv_4_b)
    conv_5 = Concatenate(name="conv5")([conv_5_1, conv_5_2])
    conv_5 = MaxPool2D(pool_size=3, strides=2, name="pool5")(conv_5)
    conv_5 = Flatten()(conv_5)

    fc6 = Dense(4096, activation='relu', name="fc6")(conv_5)
    fc6 = Dropout(rate=0.5, name="drop6")(fc6)

    fc7 = Dense(4096, activation='relu', name="fc7")(fc6)
    fc7 = Dropout(rate=0.5, name="drop7")(fc7)

    fc8 = Dense(7405, name="fc8-7k", activation='softmax')(fc7)
    
    return Model(input_, fc8)

def load_model(filepath):
    """
    Loads the model from the filepath according to the architecture defined in _create_model()

    Args:
        filepath (str): The path to the file containing the model weights
    Returns:
        The loaded model
    """
    model = _create_model()
    _load_weights(model, filepath)
    return model