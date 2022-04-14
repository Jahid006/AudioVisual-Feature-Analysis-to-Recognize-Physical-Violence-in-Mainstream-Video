



def get_backbone(backbone):
    import tensorflow as tf
    if backbone == 'DenseNet201':
        from tensorflow.keras.applications.densenet import preprocess_input
        backbone = tf.keras.applications.DenseNet201(include_top=False,weights="imagenet",pooling='avg')
    if backbone == 'InceptionV3':
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        backbone = tf.keras.applications.InceptionV3(include_top=False,weights="imagenet",pooling='avg')
    if backbone == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import preprocess_input
        backbone = tf.keras.applications.ResNet50(include_top=False,weights="imagenet",pooling='avg')
        
    return backbone, preprocess_input


def image_based_extractor(image_path,backbone):
    backbone, preprocess_input = get_backbone(backbone)
    
def mxnet_based_extractor(image_path,backbone):


def main():
    pass

if __name__ == "__main__":
    main()