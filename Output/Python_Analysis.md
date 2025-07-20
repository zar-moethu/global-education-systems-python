#### 1. Import necessary libraries


```python
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

```

#### 2. Data Collection/ Importing the dataset


```python
# Step 2.1: Load the dataset
df_edu = pd.read_excel('dataset.xlsx')

# Preview the first few rows
print(df_edu.head)

# Print the number of rows and columns in the dataset
print("Number of entries in the dataset:", df_edu.shape)

```

    <bound method NDFrame.head of           country_x code  Tertiary Percentage  ISCED5 Percentage  \
    0         Argentina  ARG            95.447912          18.103877   
    1         Australia  AUS           115.952037          25.407825   
    2           Austria  AUT            86.475597          15.080255   
    3           Belgium  BEL            80.138170           3.399620   
    4            Brazil  BRA            55.136300           0.004350   
    5            Canada  CAN            75.698825          17.629882   
    6             Chile  CHL            93.101131          25.278138   
    7             China  CHN            53.764914          22.995010   
    8          Colombia  COL            54.976519          15.692163   
    9            Cyprus  CYP            88.485329           8.480042   
    10   Czech Republic  CZE            59.274594           0.204354   
    11          Denmark  DNK            81.839762           9.341764   
    12          Estonia  EST            74.231717                NaN   
    13          Finland  FIN            92.955012                NaN   
    14           France  FRA            68.357901          13.731410   
    15          Germany  DEU            73.521058           0.240977   
    16           Greece  GRC           148.530883                NaN   
    17        Hong Kong  HKG            80.984093          19.705240   
    18          Hungary  HUN            52.444651           2.083543   
    19          Iceland  ISL            77.588839           2.646086   
    20            India  IND            28.572900                NaN   
    21          Ireland  IRL            75.179938           6.945944   
    22            Italy  ITA            66.051598           0.572927   
    23            Japan  JPN            64.617117          12.556131   
    24           Jordan  JOR            33.060867                NaN   
    25       Kazakhstan  KAZ            61.749756           8.473763   
    26           Latvia  LVA            94.864530          17.302402   
    27        Lithuania  LTU            72.008968                NaN   
    28         Malaysia  MYS            43.061606          14.089931   
    29           Mexico  MEX            42.830700           1.579458   
    30         Mongolia  MNG            68.753517           0.157899   
    31      Netherlands  NLD            87.502951           2.498367   
    32      New Zealand  NZL            80.302458          12.976771   
    33           Norway  NOR            83.230687           2.597246   
    34           Poland  POL            69.184029           0.010733   
    35         Portugal  PRT            67.930700           2.845229   
    36            Qatar  QAT            18.948995           4.710540   
    37          Romania  ROM            51.353819                NaN   
    38           Russia  RUS            86.400150          21.902220   
    39     Saudi Arabia  SAU            70.900884          10.980128   
    40  Slovak Republic  SVK            46.429019           0.820699   
    41         Slovenia  SVN            77.882773          10.829037   
    42     South Africa  ZAF            23.866558           6.932241   
    43            Spain  ESP            92.882350          19.161189   
    44           Sweden  SWE            77.329042           5.368284   
    45      Switzerland  CHE            63.308059           0.826013   
    46           Turkey  TUR           115.042059          41.863344   
    47   United Kingdom  GBR            65.773364           8.319216   
    48              USA  USA            87.888713          31.965154   
    
        ISCED6 Percentage  ISCED7 Percentage  ISCED8 Percentage  \
    0           68.238077           8.368618           0.737339   
    1           65.591820          21.327540           3.624852   
    2           40.310180          27.126033           3.959066   
    3           58.107011          15.999636           2.631904   
    4           53.314007           1.083925           0.734018   
    5           46.466291           9.183156           2.419495   
    6           60.246165           7.144492           0.432335   
    7           27.544462           2.756190           0.469252   
    8           35.608908           3.524669           0.150780   
    9           41.868006          35.507974           2.629307   
    10          38.576710          22.362087           4.443828   
    11          51.629544          18.398411           2.470043   
    12          46.041216          24.292103           3.898672   
    13          64.654877          22.540374           5.759762   
    14          27.568842          25.354663           1.702986   
    15          44.894286          23.884755           4.501040   
    16         126.987268          15.806869           5.736747   
    17          44.062660          14.308551           2.907647   
    18          33.955862          14.995659           1.409587   
    19          52.921720          19.574252           2.446781   
    20          24.835006           3.597217           0.140677   
    21          54.142605          11.295486           2.795903   
    22          39.676171          24.797628           1.004872   
    23          44.886157           5.851014           1.323816   
    24                NaN                NaN                NaN   
    25          48.897005           3.873393           0.505594   
    26          53.517502          21.534738           2.509887   
    27          52.944966          17.311583           1.752419   
    28          24.038512           3.359647           1.573517   
    29          37.962825           2.869258           0.419159   
    30          56.941652           9.813706           1.840261   
    31          67.579367          18.170783           0.000000   
    32          57.564517           6.603380           3.157878   
    33          56.087026          21.976146           2.570269   
    34          46.005814          21.268932           1.898549   
    35          38.427845          22.806852           3.850774   
    36          12.319206           1.676835           0.242414   
    37          33.186704          16.202822           1.964293   
    38          44.007432          19.097426           1.393071   
    39          55.985428           3.483220           0.452108   
    40          25.447279          17.929095           2.231946   
    41          42.377346          21.510490           3.165900   
    42          15.136176           1.301044           0.497097   
    43          54.271160          15.405076           4.044924   
    44          43.540665          25.016461           3.403704   
    45          42.712957          14.616975           5.152113   
    46          63.303589           7.935738           1.939388   
    47          42.433413          12.193517           2.827216   
    48          41.750407          12.505560           1.667592   
    
                                                country_y  year  \
    0                                           Argentina  2019   
    1                                           Australia  2019   
    2                                             Austria  2019   
    3                                             Belgium  2019   
    4                                              Brazil  2019   
    5                                              Canada  2019   
    6                                               Chile  2019   
    7                                               China  2019   
    8                                            Colombia  2019   
    9                                              Cyprus  2019   
    10                                            Czechia  2019   
    11                                            Denmark  2019   
    12                                            Estonia  2019   
    13                                            Finland  2019   
    14                                             France  2019   
    15                                            Germany  2019   
    16                                             Greece  2019   
    17     China, Hong Kong Special Administrative Region  2019   
    18                                            Hungary  2019   
    19                                            Iceland  2019   
    20                                              India  2019   
    21                                            Ireland  2019   
    22                                              Italy  2019   
    23                                              Japan  2019   
    24                                             Jordan  2019   
    25                                         Kazakhstan  2019   
    26                                             Latvia  2019   
    27                                          Lithuania  2019   
    28                                           Malaysia  2019   
    29                                             Mexico  2019   
    30                                           Mongolia  2019   
    31                                        Netherlands  2019   
    32                                        New Zealand  2019   
    33                                             Norway  2019   
    34                                             Poland  2019   
    35                                           Portugal  2019   
    36                                              Qatar  2019   
    37                                            Romania  2019   
    38                                 Russian Federation  2019   
    39                                       Saudi Arabia  2019   
    40                                           Slovakia  2019   
    41                                           Slovenia  2019   
    42                                       South Africa  2019   
    43                                              Spain  2019   
    44                                             Sweden  2019   
    45                                        Switzerland  2019   
    46                                            Türkiye  2019   
    47  United Kingdom of Great Britain and Northern I...  2019   
    48                           United States of America  2019   
    
        InternationalStudentsNO  ...  KOFFiGIdf  KOFFiGIdj  \
    0                    116330  ...         65         55   
    1                    509160  ...         81         75   
    2                     74631  ...         89         80   
    3                     52143  ...         93         86   
    4                     21803  ...         53         34   
    5                    279168  ...         87         79   
    6                     10341  ...         77         78   
    7                    201177  ...         45         47   
    8                      5249  ...         67         52   
    9                     13085  ...         97         75   
    10                    45871  ...         79         86   
    11                    32371  ...         90         83   
    12                     5043  ...         83         84   
    13                    23794  ...         91         85   
    14                   246378  ...         89         82   
    15                   333233  ...         85         80   
    16                    27789  ...         75         68   
    17                    42641  ...         98         82   
    18                    35479  ...         82         76   
    19                     1546  ...         63         58   
    20                    47424  ...         37         38   
    21                    24913  ...         98         84   
    22                    54855  ...         78         68   
    23                   202907  ...         77         80   
    24                    40544  ...         69         69   
    25                    22728  ...         79         52   
    26                     8380  ...         80         77   
    27                     6697  ...         72         73   
    28                    81953  ...         76         65   
    29                    33271  ...         61         73   
    30                     2275  ...         83         60   
    31                   108353  ...         96         84   
    32                    53002  ...         75         72   
    33                    12400  ...         91         75   
    34                    55191  ...         65         70   
    35                    35755  ...         85         81   
    36                    12332  ...         86         68   
    37                    30294  ...         56         76   
    38                   282922  ...         64         56   
    39                    73216  ...         68         55   
    40                    12730  ...         80         83   
    41                     5071  ...         76         62   
    42                    40712  ...         68         40   
    43                    77062  ...         85         79   
    44                    30912  ...         91         85   
    45                    55698  ...         96         86   
    46                   154505  ...         49         45   
    47                   489019  ...         93         86   
    48                   976562  ...         80         81   
    
        KOFSoGI_WithoutInterpersonal  InboundRatio  top_50_count  top_100_count  \
    0                           78.0      3.500110             0              1   
    1                           94.5     28.374900             5              7   
    2                           90.5     17.641230             0              0   
    3                           91.0     10.042720             0              1   
    4                           73.0      0.245040             0              0   
    5                           95.0     16.220910             3              3   
    6                           79.5      0.819470             0              0   
    7                           71.5      0.428090             3              6   
    8                           67.0      0.219050             0              0   
    9                           84.5     26.060030             0              0   
    10                          87.5     14.364180             0              0   
    11                          89.0     10.490750             0              1   
    12                          84.0     11.087370             0              0   
    13                          89.0      8.053450             0              0   
    14                          88.5      9.174700             1              3   
    15                          93.5     10.109460             0              3   
    16                          85.0      3.499400             0              0   
    17                          95.5     14.312230             3              4   
    18                          83.0     12.605300             0              0   
    19                          86.0      8.449470             0              0   
    20                          67.0      0.134930             0              0   
    21                          89.5     10.714720             0              0   
    22                          83.5      2.830840             0              0   
    23                          92.5      5.237990             2              5   
    24                          75.5     12.906060             0              0   
    25                          66.0      3.317740             0              0   
    26                          85.5     10.428720             0              0   
    27                          88.5      5.991880             0              0   
    28                          85.0      6.727130             0              1   
    29                          72.0      0.707080             0              0   
    30                          67.5      1.443300             0              0   
    31                          93.0     12.004513             0              3   
    32                          89.0     20.768790             0              1   
    33                          93.5      4.275660             0              0   
    34                          83.0      3.856860             0              0   
    35                          84.0      9.711260             0              0   
    36                          84.5     35.293780             0              0   
    37                          79.5      5.675700             0              0   
    38                          73.5      4.965320             0              1   
    39                          75.5      4.429100             0              0   
    40                          85.5      9.040620             0              0   
    41                          80.5      6.673160             0              0   
    42                          72.0      3.471960             0              0   
    43                          87.5      3.697830             0              0   
    44                          93.0      7.151650             0              1   
    45                          92.5     17.798700             2              3   
    46                          69.0      1.987110             0              0   
    47                          94.5     18.677060             8             18   
    48                          95.5      5.188870            19             33   
    
        top_500_count  top_1000_count  total_ranked_universities  \
    0               5              15                         15   
    1              25              37                         37   
    2               5               8                          8   
    3               7               8                          8   
    4               5              22                         22   
    5              18              26                         26   
    6               2              11                         11   
    7              22              40                         40   
    8               3              11                         11   
    9               0               0                          0   
    10              1               5                          5   
    11              5               5                          5   
    12              1               3                          3   
    13              7              10                         10   
    14             17              35                         35   
    15             30              45                         45   
    16              1               6                          6   
    17              6               7                          7   
    18              1               6                          6   
    19              0               0                          0   
    20              9              24                         24   
    21              5               8                          8   
    22             12              30                         30   
    23             17              44                         44   
    24              0               3                          3   
    25              5              10                         10   
    26              0               3                          3   
    27              1               4                          4   
    28              6              13                         13   
    29              2              14                         14   
    30              0               0                          0   
    31             13              13                         13   
    32              8               8                          8   
    33              4               4                          4   
    34              2              14                         14   
    35              4               7                          7   
    36              1               1                          1   
    37              0               5                          5   
    38             15              27                         27   
    39              4               8                          8   
    40              0               3                          3   
    41              0               2                          2   
    42              3               9                          9   
    43             13              25                         25   
    44              8               8                          8   
    45              8               9                          9   
    46              2              10                         10   
    47             51              76                         76   
    48             94             156                        156   
    
                           WESP  
    0                Developing  
    1                 Developed  
    2                 Developed  
    3                 Developed  
    4                Developing  
    5                 Developed  
    6                Developing  
    7                Developing  
    8                Developing  
    9                 Developed  
    10                Developed  
    11                Developed  
    12                Developed  
    13                Developed  
    14                Developed  
    15                Developed  
    16                Developed  
    17               Developing  
    18                Developed  
    19                Developed  
    20               Developing  
    21                Developed  
    22                Developed  
    23                Developed  
    24               Developing  
    25  Economies in transition  
    26                Developed  
    27                Developed  
    28               Developing  
    29               Developing  
    30               Developing  
    31                Developed  
    32                Developed  
    33                Developed  
    34                Developed  
    35                Developed  
    36               Developing  
    37                Developed  
    38  Economies in transition  
    39               Developing  
    40                Developed  
    41                Developed  
    42               Developing  
    43                Developed  
    44                Developed  
    45                Developed  
    46                Developed  
    47                Developed  
    48                Developed  
    
    [49 rows x 45 columns]>
    Number of entries in the dataset: (49, 45)


#### 3. Data Processing & Feature Engineering


```python
# Step 3.1: Features/Variables selection — keeping only the columns needed for this analysis

# List of relevant columns we want to keep for clustering and plotting
columns_of_interest = [
    'InboundRatio', 'InternationalStudentsNO',
    'KOFPoGI', 'KOFEcGI', 'KOFSoGI',                  # 🌍 Globalisation Indices (Political, Economic & Social)
    'ISCED5 Percentage', 'ISCED6 Percentage',         # 🎓 Education Levels (Short-cycle tertiary, Bachelor)
    'ISCED7 Percentage', 'ISCED8 Percentage',         # 🎓 Education Levels (Masters, Doctoral)
    'top_50_count', 'top_100_count',                  # 🏆 University rankings (Top tiers)
    'top_500_count', 'top_1000_count',                # 🏆 Lower-tier rankings
    'WESP', 'country_x'                               # 🌐 World Economic Situation and Prospects 2021 + Country name
]

# Create a new DataFrame with only selected features/variables 
df_interest = df_edu[columns_of_interest]

# Print shape (rows, columns) of the selected features/variables 
print(df_interest.shape)  

```

    (49, 15)



```python
# ✅ Step 3.2: Calculate the missing values for selected features and drop

# Count of missing values per column
missing_count = df_interest[columns_of_interest].isnull().sum()

# Percentage of missing values per column 
missing_percentage = df_interest[columns_of_interest].isnull().mean() * 100

# Combine counts and percentages into one DataFrame
missing_summary = pd.DataFrame({
    'Missing Count': missing_count,
    'Missing (%)': missing_percentage.round(2)
})

# Display the summary
print("📊 Missing values summary (by column):")
display(missing_summary)

# Drop rows with missing values in the selected columns
df_interest_cleaned = df_interest.dropna(subset=columns_of_interest)

# Display row before and after cleaning
print(f"\nOriginal number of rows: {df_interest.shape[0]}")
print(f"Number of rows after dropping missing values: {df_interest_cleaned.shape[0]}")

```

    📊 Missing values summary (by column):



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Missing Count</th>
      <th>Missing (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>InboundRatio</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>InternationalStudentsNO</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>KOFPoGI</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>KOFEcGI</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>KOFSoGI</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>ISCED5 Percentage</th>
      <td>7</td>
      <td>14.29</td>
    </tr>
    <tr>
      <th>ISCED6 Percentage</th>
      <td>1</td>
      <td>2.04</td>
    </tr>
    <tr>
      <th>ISCED7 Percentage</th>
      <td>1</td>
      <td>2.04</td>
    </tr>
    <tr>
      <th>ISCED8 Percentage</th>
      <td>1</td>
      <td>2.04</td>
    </tr>
    <tr>
      <th>top_50_count</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>top_100_count</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>top_500_count</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>top_1000_count</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>WESP</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>country_x</th>
      <td>0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
</div>


    
    Original number of rows: 49
    Number of rows after dropping missing values: 42


### 4. Exploratory Data Analysis (EDA)


```python
# Step 4.1: View Summary Statistics of the Cleaned Dataset and Selected Features

#  Basic descriptive statistics for all selected numeric columns
print("📈 Summary Statistics (Numeric Columns):\n")
summary_stats = df_interest_cleaned.describe().T.round(2)
display(summary_stats)


#  Data types for each column
print("\n🔎 Data Types:\n")
print(df_interest_cleaned.dtypes)

# Total number of missing values in the cleaned dataset
total_missing = df_interest_cleaned.isnull().sum().sum()
print(f"\n🚫 Total Missing Values in Cleaned Dataset: {total_missing}")

# Export data types
df_interest_cleaned.dtypes.to_frame(name="DataType").to_csv("Step4.1_DataTypes.csv")

# Export total missing value count to a text file
with open("Step4.1_TotalMissing.txt", "w") as f:
    f.write(f"Total Missing Values: {total_missing}")


```

    📈 Summary Statistics (Numeric Columns):
    



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>InboundRatio</th>
      <td>42.0</td>
      <td>9.37</td>
      <td>8.02</td>
      <td>0.22</td>
      <td>3.55</td>
      <td>7.80</td>
      <td>12.46</td>
      <td>35.29</td>
    </tr>
    <tr>
      <th>InternationalStudentsNO</th>
      <td>42.0</td>
      <td>117317.38</td>
      <td>183894.02</td>
      <td>1546.00</td>
      <td>22034.25</td>
      <td>49007.00</td>
      <td>114335.75</td>
      <td>976562.00</td>
    </tr>
    <tr>
      <th>KOFPoGI</th>
      <td>42.0</td>
      <td>84.95</td>
      <td>13.51</td>
      <td>29.00</td>
      <td>79.50</td>
      <td>89.50</td>
      <td>93.00</td>
      <td>98.00</td>
    </tr>
    <tr>
      <th>KOFEcGI</th>
      <td>42.0</td>
      <td>71.98</td>
      <td>12.99</td>
      <td>42.00</td>
      <td>64.00</td>
      <td>76.50</td>
      <td>82.00</td>
      <td>90.00</td>
    </tr>
    <tr>
      <th>KOFSoGI</th>
      <td>42.0</td>
      <td>79.98</td>
      <td>9.36</td>
      <td>59.00</td>
      <td>73.25</td>
      <td>83.00</td>
      <td>87.00</td>
      <td>91.00</td>
    </tr>
    <tr>
      <th>ISCED5 Percentage</th>
      <td>42.0</td>
      <td>10.63</td>
      <td>9.80</td>
      <td>0.00</td>
      <td>2.52</td>
      <td>8.48</td>
      <td>16.90</td>
      <td>41.86</td>
    </tr>
    <tr>
      <th>ISCED6 Percentage</th>
      <td>42.0</td>
      <td>45.24</td>
      <td>13.08</td>
      <td>12.32</td>
      <td>38.85</td>
      <td>44.47</td>
      <td>54.24</td>
      <td>68.24</td>
    </tr>
    <tr>
      <th>ISCED7 Percentage</th>
      <td>42.0</td>
      <td>14.23</td>
      <td>8.70</td>
      <td>1.08</td>
      <td>6.74</td>
      <td>14.81</td>
      <td>21.46</td>
      <td>35.51</td>
    </tr>
    <tr>
      <th>ISCED8 Percentage</th>
      <td>42.0</td>
      <td>2.10</td>
      <td>1.35</td>
      <td>0.00</td>
      <td>0.80</td>
      <td>2.09</td>
      <td>2.89</td>
      <td>5.15</td>
    </tr>
    <tr>
      <th>top_50_count</th>
      <td>42.0</td>
      <td>1.10</td>
      <td>3.26</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>19.00</td>
    </tr>
    <tr>
      <th>top_100_count</th>
      <td>42.0</td>
      <td>2.26</td>
      <td>5.79</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.50</td>
      <td>33.00</td>
    </tr>
    <tr>
      <th>top_500_count</th>
      <td>42.0</td>
      <td>10.21</td>
      <td>16.54</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>5.00</td>
      <td>12.75</td>
      <td>94.00</td>
    </tr>
    <tr>
      <th>top_1000_count</th>
      <td>42.0</td>
      <td>18.64</td>
      <td>26.71</td>
      <td>0.00</td>
      <td>6.25</td>
      <td>9.50</td>
      <td>24.25</td>
      <td>156.00</td>
    </tr>
  </tbody>
</table>
</div>


    
    🔎 Data Types:
    
    InboundRatio               float64
    InternationalStudentsNO      int64
    KOFPoGI                      int64
    KOFEcGI                      int64
    KOFSoGI                      int64
    ISCED5 Percentage          float64
    ISCED6 Percentage          float64
    ISCED7 Percentage          float64
    ISCED8 Percentage          float64
    top_50_count                 int64
    top_100_count                int64
    top_500_count                int64
    top_1000_count               int64
    WESP                        object
    country_x                   object
    dtype: object
    
    🚫 Total Missing Values in Cleaned Dataset: 0



```python
#Step 4.2: Visualize Education Levels by Country using a Stacked Bar Chart
percentage_cols = ['ISCED5 Percentage', 'ISCED6 Percentage', 'ISCED7 Percentage', 'ISCED8 Percentage']
df_stacked = df_interest_cleaned[['country_x'] + percentage_cols].set_index('country_x')

# Create plot
fig, ax = plt.subplots(figsize=(15, 6))

# Steel blue color palette (from dark to light shades)
colors = ['#4682B4', '#5B9BD5', '#87CEFA', '#B0C4DE']  # steel blue tones

# Plot stacked bar chart
df_stacked.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=colors)

# Aesthetics
ax.set_title("Percentage Distribution of Education Levels by Country", fontsize=15, fontweight='bold')
ax.set_ylabel("Percentage (%)", fontsize=13)
ax.set_xlabel("Country", fontsize=13)

# Tick label font size and tick mark styling
ax.tick_params(axis='x', labelsize=11, labelrotation=90)
ax.tick_params(axis='y', labelsize=11)

# Move legend outside
ax.legend(title="Education Level", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.show()

```


    
![png](output_9_0.png)
    



```python
#Step 4.3: Visualize University Rankings by Country using a Stacked Bar Chart

# Ranking columns
ranking_cols = ['top_50_count', 'top_100_count', 'top_500_count', 'top_1000_count']

# Filter valid rows
df_ranked = df[['country_x'] + ranking_cols].copy()
df_ranked = df_ranked.dropna(subset=ranking_cols, how='all')
df_ranked = df_ranked[df_ranked[ranking_cols].sum(axis=1) > 0].copy()

# Calculate percentages
df_ranked['total'] = df_ranked[ranking_cols].sum(axis=1)
for col in ranking_cols:
    df_ranked[col + '_pct'] = (df_ranked[col] / df_ranked['total']) * 100

# Sort by top_50 percentage
df_ranked = df_ranked.sort_values(by='top_50_count_pct', ascending=False)

# Steel blue color palette
colors = ['#4682B4', '#5B9BD5', '#87CEFA', '#B0C4DE']  # dark to light

# Plot
fig, ax = plt.subplots(figsize=(10, 12))
y = np.arange(len(df_ranked))
left = np.zeros(len(df_ranked))

for i, col in enumerate(ranking_cols):
    pct_col = col + '_pct'
    bars = ax.barh(y, df_ranked[pct_col], left=left, height=0.6,
                   label=col.replace('_', ' ').title(), color=colors[i])
    
    # Percentage labels
    for j, val in enumerate(df_ranked[pct_col]):
        if val > 0:
            ax.text(left[j] + val / 2, y[j], f"{val:.0f}%", ha='center', va='center_baseline',
                    fontsize=9, color='maroon')

    left += df_ranked[pct_col].values


# Axis labels and title
ax.set_yticks(y)
ax.set_yticklabels(df_ranked['country_x'], fontsize=12)
ax.set_xlabel("Percentage (%)", fontsize=12)
ax.set_title("Percentage Distribution of University Rankings by Country",
             fontsize=14, fontweight='bold')

# Tick label font size and tick mark size
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)

# Add space between bars
ax.set_ylim(-0.5, len(df_ranked) - 0.5)

# Legend outside
ax.legend(title="Ranking Tier", bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.show()

```


    
![png](output_10_0.png)
    


## 5. Data Analysis – Unsupervised Machine Learning Algorithm: Clustering Analysis


```python
# Step 5.1: K-Means Clustering - to choose the number of clusters (k), we chose to use the Elbow Method in this analysis.

#Drop non-numeric columns
X = df_interest_cleaned.drop(['country_x', 'WESP'], axis=1)

# Step 2: Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Step 3: Define the WCSS function
def wcss(x, kmax):
    wcss_s = []
    for k in range(2, kmax + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(x)
        wcss_s.append(kmeans.inertia_) 
    return wcss_s

# Step 4: Plot the elbow curve
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
kmax = 10
ax.plot(range(2, kmax + 1), wcss(X_std, kmax), marker='o')
ax.set_xlabel('Number of clusters', fontsize=13)
ax.set_ylabel('Sum of squares', fontsize=13)
ax.set_title('Sum of squared error (elbow curve) by number of clusters', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

```


    
![png](output_12_0.png)
    



```python
##Step 5.1: Data Modeling – Hierarchical Clustering with dendrogram analysis

# Step 1: Drop non-numeric columns
X = df_interest_cleaned.drop(['country_x', 'WESP'], axis=1)

# Step 2: Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Step 3: Perform hierarchical clustering
Z = linkage(X_std, method='ward')

# Step 4: Plot the dendrogram
plt.figure(figsize=(8, 7))
dend = dendrogram(
    Z,
    orientation='right',
    labels=df_interest_cleaned['country_x'].tolist()
)

plt.title('Hierarchical Clustering - Dendrogram', fontsize=14, fontweight='bold')
plt.xlabel('Distance', fontsize=12)
plt.ylabel('Country', fontsize=12)
plt.tight_layout()
plt.show()



```


    
![png](output_13_0.png)
    



```python
#Step 5.3: Cluster Visualisation with Scatter Plot

# STEP 1: Standardize numeric features for clustering
X = df_interest_cleaned.drop(['country_x', 'WESP'], axis=1)  # Exclude non-numeric or ID columns
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# STEP 2: Apply K-Means Clustering with k=2
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_std)
df_interest_cleaned['Cluster'] = kmeans.labels_  # Add cluster label to the dataset

# STEP 3: Visualize clusters using a scatter plot
plt.figure(figsize=(20, 8))

sns.scatterplot(
    x='KOFEcGI',  # Economic Globalisation Index
    y='ISCED7 Percentage',  # Participation in Master's level education
    hue='Cluster',          # Color by cluster
    data=df_interest_cleaned,
    palette=["green", "orange", "red"],
    s=100  # Dot size
)

# Set axis labels and plot title
plt.xlabel('Economic Globalisation Index (KOFEcGI)', fontsize=15)
plt.ylabel("ISCED7 Percentage: Participation in Master’s level", fontsize=15)
plt.title("Clustering Countries by Economic Globalisation and Master's-Level Participation", 
          fontsize=16, fontweight='bold')

# Extend x-axis slightly to avoid label overlap
plt.xlim(df_interest_cleaned['KOFEcGI'].min() - 2,
         df_interest_cleaned['KOFEcGI'].max() + 3)

# Tick label font size and tick mark styling
plt.tick_params(axis='x', labelsize=12, length=6, width=1.2)
plt.tick_params(axis='y', labelsize=12, length=6, width=1.2)

# Add country name labels beside each point
for i in range(len(df_interest_cleaned)):
    plt.text(
        df_interest_cleaned['KOFEcGI'].iloc[i] + 0.3,
        df_interest_cleaned['ISCED7 Percentage'].iloc[i],
        df_interest_cleaned['country_x'].iloc[i],
        fontsize=11,
        alpha=0.85,
        color='black'
    )

# Legend
plt.legend(title='Cluster', title_fontsize=14)
plt.tight_layout()
plt.show()

```


    
![png](output_14_0.png)
    



```python

```
