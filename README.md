# ASD-INTERNSHIP
This Repository contains the project done by me while working as an intern at Academy Of Skill Development
<h1> Stock Market Prediction Using Machine Learning </h1> 
<h1> Program Implementation </h1>
We shall move on to the part where we put the LSTM into use in predicting the stock value using Machine Learning in Python.

<h1> Step 1 – Importing the Libraries </h1>
As we all know, the first step is to import libraries that are necessary to preprocess the stock data of Microsoft Corporation and the other required libraries for building and visualising the outputs of the LSTM model. For this, we will use the Keras library under the TensorFlow framework. The required modules are imported from the Keras library individually.

<h1> Step 2 – Getting Visualising the Data </h1>
Using the Pandas Data reader library, we shall upload the local system’s stock data as a Comma Separated Value (.csv) file and store it to a pandas DataFrame. Finally, we shall also view the data.

<h1> Step 3 – Print the DataFrame Shape and Check for Null Values. </h1>
In this yet another crucial step, we first print the shape of the dataset. To make sure that there are no null values in the data frame, we check for them. The presence of null values in the dataset tend to cause problems during training as they act as outliers causing a wide variance in the training process.

<h1> Step 4 – Plotting the True Adjusted Close Value </h1>
The final output value that is to be predicted using the Machine Learning model is the Adjusted Close Value. This value represents the closing value of the stock on that particular day of stock market trading. 

<h1> Step 5 – Setting the Target Variable and Selecting the Features </h1> 
In the next step, we assign the output column to the target variable. In this case, it is the adjusted relative value of the Microsoft Stock. Additionally, we also select the features that act as the independent variable to the target variable (dependent variable). To account for training purpose, we choose four characteristics, which are:

  <ul>
  <li>Open</li>
  <li>High</li>
  <li>Low</li>
  <li>Volume</li>
  </ul>
  
# Step 6 – Scaling
To reduce the data’s computational cost in the table, we shall scale down the stock values to values between 0 and 1. In this way, all the data in big numbers get reduced, thus reducing memory usage. Also, we can get more accuracy by scaling down as the data is not spread out in tremendous values. This is performed by the MinMaxScaler class of the sci-kit-learn library.

# Step 7 – Splitting to a Training Set and Test Set.
Before feeding the data into the training model, we need to split the entire dataset into training and test set. The Machine Learning LSTM model will be trained on the data present in the training set and tested upon on the test set for accuracy and backpropagation.

For this, we will be using the TimeSeriesSplit class of the sci-kit-learn library. We set the number of splits as 10, which denotes that 10% of the data will be used as the test set, and 90% of the data will be used for training the LSTM model. The advantage of using this Time Series split is that the split time series data samples are observed at fixed time intervals.

# Step 8 – Processing the Data For LSTM
Once the training and test sets are ready, we can feed the data into the LSTM model once it is built. Before that, we need to convert the training and test set data into a data type that the LSTM model will accept. We first convert the training data and test data to NumPy arrays and then reshape them to the format (Number of Samples, 1, Number of Features) as the LSTM requires that the data be fed in 3D form. As we know, the number of samples in the training set is 90% of 7334, which is 6667, and the number of features is 4, the training set is reshaped to (6667, 1, 4). Similarly, the test set is also reshaped.

# Step 9 – Building the LSTM Model
Finally, we come to the stage where we build the LSTM Model. Here, we create a Sequential Keras model with one LSTM layer. The LSTM layer has 32 unit, and it is followed by one Dense Layer of 1 neuron.

We use Adam Optimizer and the Mean Squared Error as the loss function for compiling the model. These two are the most preferred combination for an LSTM model. 

# Step 10 – Training the Model
Finally, we train the LSTM model designed above on the training data for 100 epochs with a batch size of 8 using the fit function.

# Step 11 – LSTM Prediction
With our model ready, it is time to use the model trained using the LSTM network on the test set and predict the Adjacent Close Value of the Microsoft stock. This is performed by using the simple function of predict on the lstm model built.

# Step 12 – True vs Predicted Adj Close Value – LSTM
Finally, as we have predicted the test set’s values, we can plot the graph to compare both Adj Close’s true values and Adj Close’s predicted value by the LSTM Machine Learning model.
