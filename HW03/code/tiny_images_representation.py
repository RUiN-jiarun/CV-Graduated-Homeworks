import numpy as np
from PIL import Image
from utils import Dataloader, knn_classifier, visualize

def get_tiny_images(dir):
    w = h = 16                                      # resize to 16x16
    tiny_images = []

    for img_path in dir:
        img = Image.open(img_path).resize((w, h))
        img = (img - np.mean(img)) / np.std(img)    # make it zero mean and unit length
        tiny_images.append(img.flatten())

    tiny_images = np.asarray(tiny_images)

    return tiny_images

def main():
    categories = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial',
           'InsideCity', 'Kitchen', 'LivingRoom', 'Mountain', 'Office',
           'OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding']
    train_image_paths, test_image_paths, train_labels, test_labels = Dataloader('../data', categories, img_num=100)
    train_feature = get_tiny_images(train_image_paths)
    test_feature  = get_tiny_images(test_image_paths)
    result = knn_classifier(train_feature, train_labels, test_feature, k=1)
    accuracy = float(len([x for x in zip(test_labels, result) if x[0]== x[1]]))/float(len(test_labels))
    print('Average accuracy: ' + str(accuracy))
    for category in categories:
        accuracy_each = float(len([x for x in zip(test_labels, result) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
        print(str(category) + ': ' + str(accuracy_each))
    visualize('Tiny Image & KNN', train_labels, test_labels, categories, result, '../results/tiny_image_knn')

if __name__ == '__main__':
    main()