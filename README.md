# Assignment 4: Performance Metrics, and Optimisation

### Student ID: 300637212                    Student Name: Xieji Li



## Part 1: Performance Metrics in Regression [30 marks]

### Requirements

### Based on exploratory data analysis, discuss what preprocessing that you need to do before regression, and provide evidence and justifications.

- Step1. Load Data && split the dataset

  <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926164001209.png" alt="image-20220926164001209" style="zoom: 25%;" />

* Step 2. Initial Data Analysis

  <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926153442422.png" alt="image-20220926153442422" style="zoom:50%;" />

**Conclusion:** In this stage we can know there are 10 features in this dataset. We need to predict the value of price based on other 9 features. Also, there is no missing value in this dataset.

* correlation analysis

  Heat map

  ![image-20220926163800751](/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926163800751.png)

| price |          |
| :---: | :------: |
| carat | 0.921591 |
|   x   | 0.884435 |
|   y   | 0.865421 |
|   z   | 0.861249 |
| price | 1.000000 |



* Step 3. Preprocess Data && Step 4. Exploratory Data Analysis

  * First, use histogram to display features, if the feature is numeric type then plot the hist according to the value of feature. If the feature is category type then plot the hist according to the frequency of the value.

  |                            carat                             |                             cut                              |                            Color                             |                           Clarity                            |                            Depth                             |
  | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
  | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926154545627.png" alt="image-20220926154545627" style="zoom:150%;" /> | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926154611593.png" alt="image-20220926154611593" style="zoom:150%;" /> | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926154621247.png" alt="image-20220926154621247" style="zoom:150%;" /> | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926154634195.png" alt="image-20220926154634195" style="zoom:150%;" /> | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926154655938.png" alt="image-20220926154655938" style="zoom:150%;" /> |
  |                          **table**                           |                            **x**                             |                            **y**                             |                            **z**                             |                          **price**                           |
  | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926155356153.png" alt="image-20220926155356153" style="zoom:150%;" /> | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926155430092.png" alt="image-20220926155430092" style="zoom:150%;" /> | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926155445448.png" alt="image-20220926155445448" style="zoom:150%;" /> | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926155457515.png" alt="image-20220926155457515" style="zoom:150%;" /> | <img src="/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926155520347.png" alt="image-20220926155520347" style="zoom:150%;" /> |

* Remove outliers

  1. In carat plot, remove the points - carat > 2.9
  2. In depth plot, remove the points - depth > 70 || depth <= 55
  3. In table plot, remove the points - table >= 70 || table  <= 50
  4. In x plot, remove the points - x >= 9 && price >= 15000
  5. In y plot, remove the points - y >= 20 || y == 0
  6. in z plot, remove the points - z >= 6 || z <= 1

* Right(origin), Left(after removing outliers)

  ![image-20220926155650593](/Users/li/Documents/VUW/COMP309/Assignment/Assignment4/assets/image-20220926155650593.png)

* Encode categorical features based on diamond documentation

  - cut
      
      | Ideal | Predium | Very Good | Good | Fair |
      | :---: | :-----: | :-------: | :--: | :--: |
      |  100  |   80    |    60     |  40  |  20  |

  - color
      - One Hot Encode

  - clarity
      
      |  I1  | SI2  | SI1  | VS2  | VVS2 | VVS1 |  IF  |
      | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
      |  30  |  40  |  50  |  60  |  70  |  80  |  90  |

* Step 5. Build classification (or regression) models using the training data && Step 7. Assess model on the test data.

  |               Model               |                          Parameters                          |      MSE       |    RMSE     |   RSE    |    MAE     | excution time |
  | :-------------------------------: | :----------------------------------------------------------: | :------------: | :---------: | :------: | :--------: | :-----------: |
  |         linear regression         |                       positive = True                        | 1647909.22(7)  | 1283.71(7)  | 0.13(7)  | 816.89(7)  |   0.02s(2)    |
  |      k-neighbors regression       |                           Default                            | 1339014.10(6)  | 1157.16(6)  | 0.12(6)  | 554.29(6)  |   1.49s(5)    |
  |         Ridge regression          |                           Default                            | 2190847.01(9)  | 1480.15(9)  | 0.21(8)  | 848.85(8)  |   0.004s(1)   |
  |     decision tree regression      |                       Max_depth = None                       |  825284.43(4)  |  908.45(4)  | 0.06(4)  | 413.07(4)  |   0.02s(3)    |
  |     random forest regression      |                     n_estimators = 1000                      |  632325.04(2)  |  795.19(2)  | 0.05(2)  | 336.00(1)  |  1m50.00s(8)  |
  |   gradient Boosting regression    |                       Max_depth = none                       |  791343.44(3)  |  889.57(3)  | 0.06(3)  | 401.06(3)  |   17.83s(7)   |
  |          SGD regression           |                           Default                            | 2178494.94(8)  | 1475.97(8)  | 0.22(10) | 864.34(10) |   0.20s(4)    |
  |  support vector regression (SVR)  |                            C=1500                            |  998458.52(5)  |  999.23(5)  | 0.09(5)  | 524.38(5)  |  3m6.66s(9)   |
  |            linear SVR             | max_iter=50000, C = 5.0, loss = 'squared_epsilon_insensitive' ,dual = True | 2201090.06(10) | 1483.61(10) | 0.21(9)  | 848.94(9)  |   10.78s(6)   |
  | multi-layer perceptron regression |                        max_iter=5000                         |  570093.37(1)  |  755.05(1)  | 0.04(1)  | 391.20(2)  | 3m22.46s(10)  |

  ### Discussion

  