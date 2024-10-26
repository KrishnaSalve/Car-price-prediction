# Car Price Predictor

![images](https://github.com/user-attachments/assets/b30819e7-9ffc-448f-aa51-eb7a1144ad08)



Welcome to the Car Price Predictor project repository! This project focuses on predicting car prices using machine learning algorithms and techniques. Leveraging a diverse set of techniques and algorithms, this project aims to develop a predictive model that estimates the prices of various automobiles based on key features.

## Description 
The Car Price Predictor ML Model is a sophisticated machine learning model developed to accurately estimate the prices of automobiles based on a variety of features. Leveraging a diverse dataset encompassing important attributes such as car make, model, year, fuel type, transmission type, mileage, and more, this model utilizes the power of regression algorithms to provide reliable price predictions.

The ML model is trained using a robust dataset comprising a wide range of car specifications and their corresponding prices. Multiple regression techniques, including Linear Regression, Decision Trees, Random Forest, and Gradient Boosting, are explored to identify the best-performing algorithm for predicting car prices.

## Overview

Predicting car prices involves analyzing a variety of factors such as the car brand, car features, Max power, mileage, and more. Car price prediction is a significant research area in machine learning, and this project delves into the process of training a model to predict car prices accurately.


## Data
The project utilizes a dataset containing information about various car brands, including features such as Car brand, cars manufacturing year, Kilometers driven, Fuel type, Seller type, Transmission, Car owner, Car Mileage, Engine, Max power Torque, No. of seats and it's selling price. This data will be used to train and test the predictive model. This data is available on Kaggle, 

    https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset


## Methodology 
The ML model is trained using a robust dataset comprising a wide range of car specifications and their corresponding prices. Multiple regression techniques, including Linear Regression, Decision Trees, Random Forest, and Gradient Boosting, are explored to identify the best-performing algorithm for predicting car prices.
The model ingests a rich set of input features including the make and model of the car, manufacturing year, type of fuel used, transmission type, mileage, and other relevant attributes. These features are meticulously preprocessed and transformed to ensure their compatibility with the model.

## Model Structure
This repository comprises a detailed project structure to facilitate the prediction process efficiently.


### Features 
1. Data Collection and Data Cleaning.
- This Car dataset contains so many features such as Manufacturing company of car, its fuel Type, Manufacturing year, Kilometers driven, seller type, Transmission, Car owner, car mileage, engine, max power, torque, no.of seats which helps to predict car selling price. This dataset can be aquired from Kaggle. This dataset contains of 8128 records with 13 columns. 
- Before data processing we have first cleaned data where we had 221 rows with missing values which we dropped using dropna and have also dropped duplicated values which can introduce biasness and skew the distribution of data to avoid such mistakes we will drop them.

      # Droping all the Null Values from dataset
      data.dropna(inplace = True)

      # To drop all the duplicate values from dataset.
      data.drop_duplicates(inplace = True)



2. Data Preprocessing.
- Many features in our dataset such as car mileage, engine, max power which required Data Preprocessing.
- name, mileage, engine and max power column contains Numerical as well as string datatype where mileage contains 'kmpl' after float for ex. 23.4 kmpl, engine column contains 'CC' after integer value for ex. 1248 CC AND max power contains string value such as 'bhp' for ex. 75 bhp. To get Numerical values we have extracted it from whole string value.

      def get_car_brand(car):
          car = car.split(' ')[0]
          return car.strip(' ')

      data['name'] = data['name'].apply(get_car_brand)
      data['mileage'] = data['mileage'].apply(get_car_brand)
      data['engine'] = data['engine'].apply(get_car_brand)
      data['max_power'] = data['max_power'].apply(get_car_brand)

      


4. Model Training.
- Before going to Data modelling we have used column transformer, pipeline to first use feature engineering using column transformer and then use various algorithms to get most accurate prices.

        ohe = OneHotEncoder()
        ohe.fit(x[['name', 'fuel', 'seller_type', 'transmission', 'owner']])

        column_trans = make_column_transformer((OneHotEncoder(categories = ohe.categories_), ['name', 'fuel', 'seller_type', 'transmission', 'owner']), remainder = 'passthrough')

        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()

        pipe = make_pipeline(column_trans, lr)
        pipe.fit(x_train, y_train)

        ycap = pipe.predict(x_test)
        print('R_score', r2_score(y_test, ycap))
        print('MAE', mean_absolute_error(y_test, ycap))

  similary, we have used various algrithms to train our model and have also used Voting and Stacking Regression.

5. Model Evaluation.
- Here we got model performances after training the model using various algorithms. Below is the screen shot of all the algorithms we have used with there performance evaluation. Below we have algorithms with their R_score and mean absolute error respectively.
 
  ![Screenshot_20241026_104342](https://github.com/user-attachments/assets/06ff5020-aabd-49fa-96a6-4d1d7e98bd46)

Further, I have also used tried touse Hyperparameter for some algorithms which were giving use good R score, but their was major difference in R score of the algorithms. Please go through my Car Price Predictor Jupyter Notebook to understand and read more about code.


## Technologies Used / Installation
### Prerequisites

- Python 3.8 or above 
- Anaconda Navigator (not necessary)
- Other libraries: Numpy, Matplotlib, xgboost, Streamlit, seaborn


## Setup

  1. Clone the Repository:
    https://github.com/KrishnaSalve/Car-price-prediction.git

  2. Navigate to the project directory in your local system:
    cd Car-price-prediction

  3. Install required packages: 
    pip install -r requirements.txt

### Usage 

  1. Run the Streamlit Application:

          streamlit run app.py


  This will start the web application and make it accessible at  http://localhost:5000/

  2. Add Laptop Configuration:

  - Open your web browser and navigate to http://localhost:5000/
  - After you are navigated to your localhost url you will see selectebox options related to Car specification like Car Brand, Car Manufacturing year, Np. of kms Driven, Fuel Type, Seller Type, Transmission Type, Car Owner, Car Mileage(in kmpl), Car Engine(in CC), Car Max Power(in bhp), No. of Seats.
  - All the options are selectbox where you have to select the option which are mentioned in our dataset except some specifications like Car Manufacturing year, Car Mileage(in kmpl), Car Engine(in CC), Car Max Power(in bhp), and No. of Seats where you have to specify the Values by scrolling the bar at specific value.

3. View the Prediction:
- After you specify the options click on the 'Predict' button.
- The predicted price will be shown on the web page as, "Predicted Car Price for given data would be Rs. {predicted price}"

4. Interpreting the result:
- Based on the specified configuration your model will predict the price of the Car.


### Result
The Car Price Predictor ML Model demonstrates strong predictive accuracy, making it a reliable tool for estimating car prices. After rigorous training, evaluation, and model comparison, the best model is selected based on its performance metrics, including the highest R-squared (R²) score and lowest Mean Absolute Error (MAE). These key performance indicators underscore the model's ability to accurately predict car prices based on a wide range of input features.
The model has undergone extensive validation and showcases excellent performance, showcasing its ability to offer precise predictions, benefiting both consumers and industry professionals. Its successful deployment as an interactive web application further solidifies its practical applicability, providing users with a seamless and accessible platform for obtaining estimated car prices. Note this Machine Learning model is only for Education purpose, do not use it for live application or practical purpose to predict car prices.
This project aligns with the overarching goal of predicting car prices using machine learning, contributing to the research and development efforts within this domain.

**Happy Predicting!** 🚗📈


Qualitative Results:

The models performance on the test dataset is as follows:

|Metric |   Value|
|-|-|
|Mean Absolute Error(mae)| 0.151
|R score| 93.5%

### Contributing
We welcome contributions from the community to help improve and expand Laptop price predictor project. If you're interested in contributing, please follow these guidelines:

**Report Issues** : 

If you encounter any bugs, errors, or have suggestions for improvements, please open an issue on the project's GitHub repository. When reporting an issue, provide a clear and detailed description of the problem, including any relevant error messages or screenshots.

**Submit Bug Fixes or Enhancements** : 

If you've identified a bug or have an idea for an enhancement, feel free to submit a pull request. Before starting work, please check the existing issues to ensure that your proposed change hasn't already been addressed.
When submitting a pull request, make sure to:

    1. Fork the repository and create a new branch for your changes.
    2. Clearly describe the problem you're solving and the proposed solution in the pull request description.
    3. Follow the project's coding style and conventions.
    4. Include relevant tests (if applicable) to ensure the stability of your changes.
    5. Update the documentation, including the README file, if necessary.


**Improve Documentation**

If you notice any issues or have suggestions for improving the project's documentation, such as the README file, please submit a pull request with the necessary changes.

**Provide Feedback**

We value your feedback and suggestions for improving the Laptop Price Predictor project. Feel free to share your thoughts, ideas, or use cases by opening an issue or reaching out to the project maintainers.


### Contact

If you have any questions, feedback, or would like to get in touch with the project maintainers, you can reach us through the following channels:

- **Project Maintainer**

  Name : Krishna Salve 

  Email : krishnasalve97@gmail.com

  Linkedin : Krishna Salve

  GitHub : KrishnaSalve


- **Project Repository**

      https://github.com/KrishnaSalve/Car-price-prediction