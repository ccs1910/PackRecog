from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import os


class test():
    def __init__(self, *args, **kwargs):

        print("Initiate Testing ...  ")

        self.model = load_model('cigarettes2.h5')
        self.model.summary()     

        base_dir = '/Users/csantoso/AppData/Local/My Private Documents/_BitBucket/DL_python/pr/PackRecog/rokok_small'

        test_dir = os.path.join(base_dir, 'test')


        test_datagen = ImageDataGenerator(rescale=1./255)

        test_generator = test_datagen.flow_from_directory(
            test_dir, target_size=(150, 150),
            batch_size=20
        )

        self.labels_dict = test_generator.class_indices

        print("test_generator.class:", self.labels_dict)

        test_loss, test_acc = self.model.evaluate_generator(
            test_generator
        )

        print("test_loss:", test_loss, "\n\n\n")

        print("test_acc:", test_acc,"\n\n\n")



if __name__ == "__main__":

    test()
