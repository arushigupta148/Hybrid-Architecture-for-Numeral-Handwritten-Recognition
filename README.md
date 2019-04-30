# Hybrid-Architecture-for-Numeral-Handwritten-Recognition

Designed a hybrid architecture using a Random Forest Classifier and a Multi-Layer Perceptron to classify handwritten digits, taken from the MNIST dataset. Published in the 5th International Conference on Future Computational Technologies (ICFCT), Kyoto, Japan (April 2017).

The system allowed false predictions of only 0.97% using HC. The Random Forest and Multi-Layer Perceptron are used specifically as they have shown high accuracies individually for the dataset, that is, 97.06% and 97.87% respectively.

Data preprocessing- The dataset contains 60,000 samples for training data and 10,000 samples for testing data, with each sample having dimensions 28x28.
Following preprocessing techniques were applied- 
1)	Otsu’s method: The images obtained from the MNIST database are in grayscale with each pixel value ranging from 0 to 255. This range of values provided various levels of the colour grey, leading to a blur forming in the images. After the application of Otsu’s binarization, every pixel was represented by either a 0 or 1 (binary values). 
2)	Removal of low variance features: Each image from the MNIST dataset has dimensions 28x28, resulting in 784 pixels where every pixel is being considered as a feature. However, in these 784 features, not all of them contribute to improve the analysis. There are some pixels which always have the value 0 (low variance), which may reduce the quality of analysis (as they may be assumed as similarity). By eliminating these pixels as features, the total number of features were reduced to 641. 
After applying this on the training data, the same techniques were applied on the testing dataset.

Classification
RFCs are an ensemble of decision trees which predict the class of an item as the mode of all predictions from their respective ensemble. The RFC chosen consisted of 500 estimators, that is, 500 decision trees which were trained using 60,000 training samples. Each of these trees were generated using Gini index ( Gini coefficient of zero expresses perfect equality, 1 (or 100%) expresses maximal inequality among values- used to decide which feature to use next on a node) to assess the purity of nodes. The training data provided to each decision tree is different using bagging which helps to avoid overfitting of data. 

The MLP chosen consists of one hidden layer consisting of 1500 neurons. Each pixel is given as input to these neurons and after 500 iterations of training with a learning rate of 1x10**-7 using backpropagation, the logistic function (F(X)=1/1+E-X) (sigmoid function) is used as an activation function to get values between 0 and 1.

Initially, the testing data is passed to the RFC and MLP individually and the result is observed. After completion of individual testing, the testing data is passed to the HC, which sends the images one by one to the RFC and MLP. For every image, the HC compares the outputs from both classifiers. If the outputs match, it makes a decision to provide the prediction, else it gives no output. 

 

The challenge faced during the implementation of HC was arbitrary results from RFC. With every generation of an RFC model, the accuracy varied between 96.92% and 97.08%. Upon combination with MLP, a range of accuracies obtained for HC were between 98.89% to 99.03%. This is accounted due to bagging, as the process assigns a random and unique training set to every decision tree in the RFC each time.
It was found that these models recognised the training data with a 100% accuracy. When supplied with the testing data, the RFC had an accuracy of 97±0.08%.

