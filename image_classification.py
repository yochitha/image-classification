import numpy as np
import cv2
import os

from utility import WeakClassifier, VJ_Classifier

def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    X = np.empty((0, size[0] * size[1]), float)
    y = np.empty(len(images_files), float)
    i = 0

    for img in images_files:
        image = cv2.cvtColor(cv2.imread(os.path.join(folder, img)), cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, size)
        X = np.append(X, np.array([image.flatten().astype(float)]), axis=0)
        y[i] = int(img[8])
        i += 1

    return X, y

def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    y = np.reshape(y, (len(y), 1))
    conc_list = np.concatenate((y, X), axis=1)

    shuffle_list = np.random.permutation(conc_list)
    i = int(np.floor(p * X.shape[0]))

    train_set = shuffle_list[0:i]
    test_set = shuffle_list[i:]
    Xtrain = train_set[:, 1:]
    ytrain = train_set[:, 0]
    Xtest = test_set[:, 1:]
    ytest = test_set[:, 0]

    return Xtrain, ytrain, Xtest, ytest

def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    mean_face = np.mean(x, axis=0)

    return mean_face

def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """

    mu = get_mean_face(X)
    X = X - mu
    L = np.dot(np.transpose(X), X)
    w, v = np.linalg.eigh(L)

    eigen_vectors = np.empty((0, len(L)), float)
    eigen_values = np.empty(k, float)
    j = 0

    for i in range(len(L) - 1, len(L) - k - 1, -1):
        eigen_vectors = np.append(eigen_vectors, np.array([v[:, i]]), axis=0)
        eigen_values[j] = w[i]
        j = j + 1

    return eigen_vectors.transpose(), eigen_values


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        for i in range(self.num_iterations):
            norm_weights = self.weights / np.sum(self.weights)
            weak_classifier = WeakClassifier(self.Xtrain, self.ytrain, norm_weights)
            weak_classifier.train()
            wk_result = [weak_classifier.predict(x) for x in self.Xtrain]
            self.weakClassifiers.append(weak_classifier)

            error = np.sum(self.weights[wk_result != self.ytrain])
            alpha = 0.5 * np.log((1 - error) / error)
            self.alphas.append(alpha)

            if error > self.eps:
                self.weights = self.weights * np.exp(-self.ytrain * alpha * wk_result)
            else:
                break

        return None

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        y_pred = self.predict(self.Xtrain)
        good = 0
        bad = 0
        for i in range(len(self.Xtrain)):
            if y_pred[i] == self.ytrain[i]:
                good += 1
            else:
                bad += 1

        return good, bad

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """

        prod = []
        for i in range(len(self.alphas)):
            y_pred = [self.weakClassifiers[i].predict(x) for x in X]
            y_pred = np.array(y_pred)
            prod.append(self.alphas[i] * y_pred)

        predict_values = np.sign(np.sum(prod, axis=0))

        return predict_values


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        image = np.zeros(shape, dtype='uint8')

        image[self.position[0]: self.position[0] + self.size[0] // 2,
        self.position[1]: self.position[1] + self.size[1]] = 255

        image[self.position[0] + self.size[0] // 2: self.position[0] + self.size[0],
        self.position[1]: self.position[1] + self.size[1]] = 126

        return image

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        image = np.zeros(shape, dtype='uint8')

        image[self.position[0]: self.position[0] + self.size[0],
        self.position[1]: self.position[1] + self.size[1] // 2] = 255

        image[self.position[0]: self.position[0] + self.size[0],
        self.position[1] + self.size[1] // 2: self.position[1] + self.size[1]] = 126

        return image

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        image = np.zeros(shape, dtype='uint8')

        image[self.position[0]: self.position[0] + self.size[0] // 3,
        self.position[1]: self.position[1] + self.size[1]] = 255

        image[self.position[0] + self.size[0] // 3: self.position[0] + 2 * (self.size[0] // 3),
        self.position[1]: self.position[1] + self.size[1]] = 126

        image[self.position[0] + 2 * (self.size[0] // 3): self.position[0] + self.size[0],
        self.position[1]: self.position[1] + self.size[1]] = 255

        return image

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        image = np.zeros(shape, dtype='uint8')

        image[self.position[0]: self.position[0] + self.size[0],
        self.position[1]: self.position[1] + self.size[1] // 3] = 255

        image[self.position[0]: self.position[0] + self.size[0],
        self.position[1] + self.size[1] // 3: self.position[1] + 2 * (self.size[1] // 3)] = 126

        image[self.position[0]: self.position[0] + self.size[0],
        self.position[1] + 2 * (self.size[1] // 3): self.position[1] + self.size[1]] = 255

        return image

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        image = np.zeros(shape, dtype='uint8')

        image[self.position[0]: self.position[0] + self.size[0] // 2,
        self.position[1]: self.position[1] + self.size[1] // 2] = 126

        image[self.position[0] + self.size[0] // 2: self.position[0] + self.size[0],
        self.position[1]: self.position[1] + self.size[1] // 2] = 255

        image[self.position[0] + self.size[0] // 2: self.position[0] + self.size[0],
        self.position[1] + self.size[1] // 2: self.position[1] + self.size[1]] = 126

        image[self.position[0]: self.position[0] + self.size[0] // 2,
        self.position[1] + self.size[1] // 2: self.position[1] + self.size[1]] = 255

        return image

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        feat_type = self.feat_type
        position = self.position
        size = self.size

        if feat_type == (2, 1):
            A = ii[position[0] - 1 + size[0] // 2, position[1] + size[1] - 1] \
                - ii[position[0] - 1, position[1] + size[1] - 1] - ii[position[0] - 1 + size[0] // 2, position[1] - 1] \
                + ii[position[0] - 1, position[1] - 1]
            B = ii[position[0] + size[0] - 1, position[1] + size[1] - 1] \
                - ii[position[0] - 1 + size[0] // 2, position[1] + size[1] - 1] - ii[
                    position[0] + size[0] - 1, position[1] - 1] \
                + ii[position[0] - 1 + size[0] // 2, position[1] - 1]

            score = A - B

        elif feat_type == (1, 2):
            A = ii[position[0] - 1 + size[0], position[1] - 1 + size[1] // 2] \
                - ii[position[0] - 1, position[1] - 1 + size[1] // 2] - ii[position[0] - 1 + size[0], position[1] - 1] \
                + ii[position[0] - 1, position[1] - 1]
            B = ii[position[0] + size[0] - 1, position[1] + size[1] - 1] \
                - ii[position[0] - 1, position[1] + size[1] - 1] - ii[
                    position[0] + size[0] - 1, position[1] - 1 + size[1] // 2] \
                + ii[position[0] - 1, position[1] - 1 + size[1] // 2]

            score = A - B

        elif feat_type == (3, 1):
            A = ii[position[0] - 1 + size[0] // 3, position[1] + size[1] - 1] \
                - ii[position[0] - 1, position[1] - 1 + size[1]] - ii[position[0] - 1 + size[0] // 3, position[1] - 1] \
                + ii[position[0] - 1, position[1] - 1]
            B = ii[position[0] + 2 * (size[0] // 3) - 1, position[1] + size[1] - 1] \
                - ii[position[0] - 1 + size[0] // 3, position[1] + size[1] - 1] - ii[
                    position[0] + 2 * (size[0] // 3) - 1, position[1] - 1] \
                + ii[position[0] - 1 + size[0] // 3, position[1] - 1]
            C = ii[position[0] - 1 + size[0], position[1] + size[1] - 1] \
                - ii[position[0] - 1 + 2 * (size[0] // 3), position[1] + size[1] - 1] - ii[
                    position[0] - 1 + size[0], position[1] - 1] \
                + ii[position[0] - 1 + 2 * (size[0] // 3), position[1] - 1]
            score = A - B + C

        elif feat_type == (1, 3):
            A = ii[position[0] - 1 + size[0], position[1] - 1 + size[1] // 3] \
                - ii[position[0] - 1, position[1] - 1 + size[1] // 3] - ii[position[0] - 1 + size[0], position[1] - 1] \
                + ii[position[0] - 1, position[1] - 1]
            B = ii[position[0] + size[0] - 1, position[1] + 2 * (size[1] // 3) - 1] \
                - ii[position[0] - 1, position[1] + 2 * (size[1] // 3) - 1] - ii[
                    position[0] + size[0] - 1, position[1] - 1 + size[1] // 3] \
                + ii[position[0] - 1, position[1] - 1 + size[1] // 3]
            C = ii[position[0] - 1 + size[0], position[1] + size[1] - 1] \
                - ii[position[0] - 1, position[1] + size[1] - 1] - ii[
                    position[0] - 1 + size[0], position[1] - 1 + 2 * (size[1] // 3)] \
                + ii[position[0] - 1, position[1] - 1 + 2 * (size[1] // 3)]

            score = A - B + C

        else:
            A = ii[position[0] - 1 + size[0] // 2, position[1] - 1 + size[1] // 2] \
                - ii[position[0] - 1, position[1] - 1 + size[1] // 2] - ii[position[0] - 1 + size[0] // 2, position[1] - 1] \
                + ii[position[0] - 1, position[1] - 1]
            B = ii[position[0] + size[0] - 1, position[1] - 1 + size[1] // 2] \
                - ii[position[0] - 1 + size[0] // 2, position[1] - 1 + size[1] // 2] - ii[
                    position[0] + size[0] - 1, position[1] - 1] \
                + ii[position[0] - 1 + size[0] // 2, position[1] - 1]
            C = ii[position[0] - 1 + size[0] // 2, position[1] - 1 + size[1]] \
                - ii[position[0] - 1, position[1] - 1 + size[1]] - ii[
                    position[0] - 1 + size[0] // 2, position[1] - 1 + size[1] // 2] \
                + ii[position[0] - 1, position[1] - 1 + size[1] // 2]
            D = ii[position[0] + size[0] - 1, position[1] - 1 + size[1]] \
                - ii[position[0] - 1 + size[0] // 2, position[1] - 1 + size[1]] - ii[
                    position[0] + size[0] - 1, position[1] - 1 + size[1] // 2] \
                + ii[position[0] - 1 + size[0] // 2, position[1] - 1 + size[1] // 2]

            score = -A + B + C - D

        return score

def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    integral_img = []
    for i in range(len(images)):
        images[i] = images[i].astype(float)
        cum_sum = np.cumsum(np.cumsum(images[i], 0), 1)
        integral_img.append(np.array(cum_sum))

    return integral_img

    raise NotImplementedError


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """

    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1 * np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei, sizej]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                2 * len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                2 * len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        for i in range(num_classifiers):
            norm_weights = weights / np.sum(weights)

            vj_classifier = VJ_Classifier(scores, self.labels, norm_weights)
            vj_classifier.train()
            vj_result = [vj_classifier.predict(x) for x in scores]
            self.classifiers.append(vj_classifier)

            ei = np.where(vj_result != self.labels, 1, -1)
            error = vj_classifier.error

            beta = error / (1 - error)
            weights = weights * (beta ** (1 - ei))
            alpha = np.log(1 / beta)
            self.alphas.append(alpha)

        return None

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)
        scores = np.zeros((len(ii), len(self.haarFeatures)))
        result = []

        for clf in self.classifiers:
            feature = clf.feature
            for i in range(len(ii)):
                scores[i, feature] = self.haarFeatures[feature].evaluate(ii[i])

        prod = []
        for i in range(len(self.alphas)):
            y_pred = [self.classifiers[i].predict(x) for x in scores]
            y_pred = np.array(y_pred)
            prod.append(self.alphas[i] * y_pred)

        alpha = 0.5 * np.sum(self.alphas)
        product = np.sum(prod, axis=0)
        result = np.where(product >= alpha, 1, -1)

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_list = []
        r, c = (24, 24)
        x, y = 0, 0
        xy_coords = []

        for row in range(0, img.shape[0] - r - 6, 6):
            for col in range(0, img.shape[1] - c, 4):
                imgs = img[y:r + row, x:c + col]
                x += 4
                img_list.append(imgs)
                xy_coords.append([x, y])
            x = 0
            y += 6

        result = self.predict(img_list)
        values = np.where(result == 1)

        average_x = 0
        average_y = 0
        for i in values[0]:
            average_x += xy_coords[i][0]
            average_y += xy_coords[i][1]

        final_x = average_x // len(values[0])
        final_y = int(np.ceil(average_y / len(values[0])))

        cv2.rectangle(image, (final_x, final_y), (final_x + c - 1, final_y + r - 1), (255, 0, 0))

        cv2.imwrite("output/{}.png".format(filename), image)

        return None
