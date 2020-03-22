import numpy as np
import time
from engram.procedural import models
import tensorflow as tf

def train(model_type='CNN',in_matrix=None,labels=None):

    shape = np.asarray(in_matrix).shape[1:]
    print('Input Size: '+ str(shape))
    model = models.select(model=model_type,shape=shape)

    trials = len(in_matrix)
    channels = len(in_matrix[0])
    times = len(in_matrix[0][0])
    dim3 = len(in_matrix[0][0][0])
    reshape = (-1, channels, dim3)

    # get categories
    categories = list(labels)

    # splitting data into training and testing indices
    indices = np.arange(len(labels[categories[0]])).astype('int')
    if len(indices)%2 != 0:
        indices = indices[0:-1]
    np.random.shuffle(indices)
    train_inds,test_inds = np.split(indices,2)

    y_train_bin = []
    y_val_bin = []

    # Getting training data
    X_train = [ in_matrix[i] for i in train_inds ]
    for cat in categories:
        if not y_train_bin:
            y_train_bin = [labels[cat][train_inds]]
        else:
            y_train_bin.append(labels[cat][train_inds])
  
    # Getting test data
    X_val = [ in_matrix[i] for i in test_inds ]
    for cat in categories:
        if not y_val_bin:
            y_val_bin = [labels[cat][test_inds]]
        else:
            y_val_bin.append(labels[cat][test_inds])

    # splitting data into training and testing indices
    train_ds = create_dataset(X_train, y_train_bin)
    val_ds = create_dataset(X_val, y_val_bin)

    EPOCHS = 10
    batch_size = 32
    history = model.fit(train_ds, 
                        epochs=EPOCHS, 
                        validation_data=create_dataset(X_val, y_val_bin))
    print(history)
    MODEL_NAME = f"models/{round(history.history['accuracy'][-1]*100,2)}-epoch-{history.epoch[-1]}--loss-{round(history.history['loss'][-1],2)}.model"
    model.save(MODEL_NAME)
    print("saved:")
    print(MODEL_NAME)

    training_params = {}
    training_params['categories'] = categories
    training_params['train_inds'] = train_inds
    training_params['train_inds'] = test_inds

    return model, training_params


def create_dataset(features=None,labels_for_categories=None):
    """Load and parse dataset.
    Args:
        filenames: list of image paths
        labels: numpy array of shape (BATCH_SIZE, N_LABELS)
        is_training: boolean to indicate training mode
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE # Adapt preprocessing and prefetching dynamically to reduce GPU and CPU idle time
    SHUFFLE_BUFFER_SIZE = 1024 # Shuffle the training data by a chunck of 1024 observations

    labels = []
    for label in labels_for_categories:
        labels.append(label)
    
    labels = np.array(labels).T
    
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # # Parse and preprocess observations in parallel
    # dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)
    
    # This is a small dataset, only load it once, and keep it in memory.
    dataset = dataset.cache()
    # Shuffle the data each buffer size
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(len(labels))
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset



def get_data(features=None,labels_for_categories=None):
    training_data = {}
    categories = []

    # for label in range(len(labels)):
    #     training_data[label].append(features[:,ii,:])

    # # Begin focusing on specific categories 

    # lengths = [len(training_data[category]) for category in categories]
    # print(lengths)

    # print('Not proper derivation of validation dataset')
    # for category in categories:
    #     np.random.shuffle(training_data[category])
    #     training_data[category] = training_data[category][:min(lengths)]

    # lengths = [len(training_data[category]) for category in categories]
    # print(lengths)

    # creating X, y 
    combined_data = []
    for labels in labels_for_categories:
        for data in features:
            combined_data.append([data, labels])


    np.random.shuffle(combined_data)
    print("length:",len(combined_data))

    return combined_data, categories

# def parse_function(filename, label):
#     """Function that returns a tuple of normalized image array and labels array.
#     Args:
#         filename: string representing path to image
#         label: 0/1 one-dimensional array of size N_LABELS
#     """
#     # Read an image from a file
#     image_string = tf.io.read_file(filename)
#     # Decode it into a dense vector
#     image_decoded = tf.image.decode_jpeg(image_string, channels=CHANNELS)
#     # Resize it to fixed shape
#     image_resized = tf.image.resize(image_decoded, [IMG_SIZE, IMG_SIZE])
#     # Normalize it from [0, 255] to [0.0, 1.0]
#     image_normalized = image_resized / 255.0
#     return image_normalized, label