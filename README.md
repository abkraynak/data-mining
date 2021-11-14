# Mining the Stack Overflow Developer Survey
A prototype data mining application to compare the accuracy of decision tree and random forest regression models to 
predict annual compensation of tech workers in the US and Europe.

## Objectives


## Usage
To run, download the repository and execute the file `main.py` in the `src` directory with your python path variable. 
For example, `python3 main.py`.

## Dependencies
- python 3.8.1 and up
- pandas 1.3.4 and up
- matplotlib 3.4.3 and up
- numpy 1.21.0 and up
- sklearn 1.0.1 and up

## Methodology
### Preprocessing
The original data set provided by Stack Overflow contained 48 attribute columns and 83439 data records. Due to the 
large size of the data set, we wanted to narrow our focus to a certain subset of the data. In the preprocessing of the 
original data file, we decided to discard any records that were not employed full-time in the technology industry. Any 
record that did not contain country, converted annual salary, or yeared coded was also discarded, as this data is vital 
to our model. We also discarded some of the columns from the original data set that were open-ended. Out of the records 
that fit our requirements, we exported them to two output csv files. Records of United States data were put together in 
one output file, and records of European countries were put in the other. Data from any other countries were discarded. 
Once we have the two cleaned files, we applied additional preprocessing techniques. Any missing attributes that 
remained were replaced with 'NA' if the attributes were nominal. Two special cases existed in the columns for years 
coded and years coded professionally. Most contained a numerical value for the years, but some had a string for 'Less 
than one year' and 'More than 50 years'. These strings were replaced with 0 and 50, respectively, to keep these columns 
numerical. With these preprocessing steps complete, the data files are now ready to be processed to generate the models.

### Models
We evaluated a variety of data mining models and algorithms to find the ones that would make the most sense for our 
data set and objectives. With our goal of predicting a numerical value for annual salary, we knew we needed to use a 
compatible regression model. We found regression models for decision trees and random forests and wanted to compare 
their accuracy. We wanted to see how the accuracy of a single decision tree compares to the accuracy of a random forest 
model, which is a number of trees together. The results are detailed in the results and analysis section. Below are the 
implementation details of each model.

#### Decision tree model
We selected the DecisionTreeRegressor model from the Scikit Learn machine learning package. In order to get the most 
accurate model, we trained several models with different parameters and selected the one with the highest accuracy to 
validate. The parameter we changed was the maximum depth level of each tree. Additional factors that affect the model 
are the testing split percentage and the cross validation folds. For our models, we used 20% of the data as testing and 
80% as training and a cross validation value of 10. Out of every combination we tried, we found that a maximum depth of 
____ADD RES HERE____ resulted in the most accurate decision tree model. The accuracy of the model was ____ADD RES 
HERE____. This model will output the tree itself, several statistics of the model such as R-squared, mean absolute 
error, and mean squared error, and the ten attributes that have the largest weight in determining the result. With the 
best model selected, we then validated it against the testing data set. These steps of model generation were done for 
both the US data and the European data.

#### Random forest model
We selected the RandomForestRegressor model from the Scikit Learn machine learning package. In order to get the most 
accurate model, we trained several models with different parameters and selected the one with the highest accuracy to 
validate. The parameters we changed were the number of trees to estimate with and the maximum depth level of each tree. 
Additional factors that affect the model are the testing split percentage and the cross validation folds. For our 
models, we used 20% of the data as testing and 80% as training and a cross validation value of 10. Out of every 
combination we tried, we found that ____ADD RES HERE____ trees in the forest with a maximum depth of ____ADD RES 
HERE____ resulted in the most accurate random forest model. The accuracy of the model was ____ADD RES HERE____. This 
model will output the tree itself, several statistics of the model such as R-squared, mean absolute error, and mean 
squared error, and the ten attributes that have the largest weight in determining the result. With the best model 
selected, we then validated it against the testing data set. These steps of model generation were done for both the US 
data and the European data.

## Results and Analysis


## Authors
- Andrew Kraynak ([LinkedIn](https://www.linkedin.com/in/abkraynak/), [Github](https://github.com/abkraynak))
- Samuel Kaczynski ([LinkedIn](https://www.linkedin.com/in/samuel-kaczynski-425926196/), [Github](https://github.com/SKalltheway))