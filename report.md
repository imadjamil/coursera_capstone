# Introduction
> Buisness problem description

Traffic accident is a wolrdwide major public health concern. In 2017, 115 deaths per million inhabitants were caused by raod accident in the USA. Currently, road crashes are the leading cause of death in the USA. 

One of the most important steps in managing traffic is estimating the severity of an accident. The latter provides critical information for emergency responders to evaluate the gravity level of each situation. Severity analysis helps also in estimating the potential impact of accidents and implementing efficient accident management and avoidance procedures. Furthermore, the accdident severity analysis helps in better managing traffic congestion and minimizing the delay caused by an accident. Finally and most importantly, severity analysis helps reducing the number of traffic accidents and their impact on public health.

This work aims at deriving a machine learning model to predict the severity of a car accident in terms of the impact on traffic delay. This will help authorities to implement efficient procedures to minimise the impact of road accidents. 

# The Data Source
> Description of the dataset

The dataset used in this work is a countrywide car accident dataset that covers 49 states of the USA. The data is collected between February 2016 and June 2020. The collection was done by a variety of entities such as the US and state departments of transportation, law enforcement agencies, traffic cameras, and traffic sensors. The dataset includes about 3.5 million entries.

The dataset consists of 48 features. Each record has an unique identifier. A column indicates the source entity that collected the data record. The start time and end time of the accident are recorded as well as the starting and ending points latitude and latitude.
These information clearly help predicting the traffic delay that could be caused by the accident. For instance, an accident that occurs on a main highway in the rush hour is likely to result in a longer traffic delay than an accident that occurs in the countryside at midnight. Below is the list of columns in the dataset.

For more details about the dataset, the reader is referred to [this page](https://smoosavi.org/datasets/us_accidents).

This work uses the afformentioned dataset to predict the severity of the impact of an accident in terms of traffic delay.

# Methodology
## Data preperation
### Data shape
Before any data processing, the data has a shape equal to `(3513617, 49)`.
### The target
The target, i.e. label, column is named `Severity`. All the dataset entries have Severity values. The latter has 4 different values: 1, 2, 3, and 4, where 1 is the least impact on traffic and 4 is the most impact on traffic. 

While almost 67% of the dataset entries have Severity of 2, less than 1% of entries have Severity of 1.

The dataset is clearly skewed, and this will need some imbalance processing. I chosed to down-sample all the target classes to the minority class, i.e. Severity of 1.

### Collecting entity
Almost 69% of the entries are collected from MapQuest, ~29% are collected from Bing, and less 2% of entries have source as MapQuest-Bing.

### Datetime
> How to use the `Start_Time` column

The `Start_Time` column is very interesting because, at leat theoritically, the time when the accident occured may determine the severity impact of that accidient. In order to efficiently exploit this column, I decided to decompose it into multiple columns as follows.

- year
- month
- day
- day
- weekday

A further step that could be done is to correlate the date with holidays and important events like elections, festivals, riots, etc. But this has not been done within this project due to time constraints.

### Categorical features

## Feature selection
> What features are selected for the learning phase and how they were selected

### Uninteresting columns
The following features were not concidered because they don't bring information about the Severity or because they are redundant with other columns.

```python
['ID', 'Source', 'Timezone', 'Airport_Code', 'Zipcode', \
'Weather_Timestamp', 'Country', 'State', 'Description', 'City', 'Astonomical_Twilight', 'Notical_Twilight']
```

### Missing values
An important process that guides the feature selection is the resolving the missing values. Below is the heatmap of all the missing values in the entire dataset where a yellow color denotes a missing value.
![missing-heatmap](./img/missing_heatmap.png "Missing Values Heatmap")

Firstly, I started to look into the columns having more than 20% of missing values, this gave the following.

```
na in TMC: 29.4%
na in End_Lat: 70.5%
na in End_Lng: 70.5%
na in Number: 64.4%
na in Wind_Chill(F): 53.2%
na in Precipitation(in): 57.6%
```
My decision was to completely drop these columns since they won't give any benefit with this level of missing values.

Then, I looked into the rest of the columns still having missing values that looked as follows.

```
na in Temperature(F): 1.9%
na in Humidity(%): 2%
na in Pressure(in): 1.6%
na in Visibility(mi): 2.1
na in Wind_Speed(mph): 12.9%
na in Wind_Direction: 1.7%
na in Weather_Condition: 2.2%
```

>>>TODO: complete the list above.

For the above columns, I decided to fill the missing values as follows:
- with the column mean for `['Temperature(F)', 'Humidity(%)', 'Pressure(in)', \
'Visibility(mi)', 'Wind_Speed(mph)']` since they are of numerical type.
- with the previous value for `['Weather_Condition', 'Wind_Direction', \
'Astonomical_Twilight', 'Notical_Twilight', 'Civil_Twighlight', 'Sunrise_Sunset']` that are of categorical type. Since theoretically, the same location will have approximately the same weather and twilight conditions, this filling step has been done after sorting values with by the location of the accident using the following columns `['Start_Time', 'County', 'City']`.

At this point, there are no missing values left in the dataset.

### Final List

Following is the final list of features that where selected for the machine learning process.

```python
features=[
'Start_Lat', 'Start_Lng', 'Distance(mi)', 'Side', 'Temperature(F)', \
'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', \
'Sunrise_Sunset', 'Civil_Twilight', 'Weather_Condition', 'Crossing', \
'Junction', 'Traffic_Signal', 'year', 'month', 'day', 'hour', 'weekday'\
]
```

At this point the shape of the dataset is `(3513617, 20)`.

## Preprocessing

A `onehotencoder` is applied on the columns of categorical type.

A `standardscaler` is applied on the columns of numerical type.

After this processing, the shape of the dataset chnaged to ``.

>>>TODO fill the shape above.

## Modeling

Since the target is more of a categorical type, a classification algorithm will be the choice for the machine learning phase.

First, I started with the `RandomForestClassifier` model.

In order to get the best results, the choice was to compare the classification performance of multiple models.


# Results Comparaison

# Disscussion and Conclusion

