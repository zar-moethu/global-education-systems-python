# Global Higher Education Clustering with Python: Clustering Analyis Algorithm


### Project Overview
This project aims to understand how international student mobility is related to the level of globalisation and the structure of higher education systems across countries in 2019.
We apply unsupervised learning, specifically clustering analysis, to group countries based on similarities in:

**Education system characteristics**, including:  
- ISCED Participation Levels (Levels 5 to 8)
- QS World University Rankings (Top 50, 100, 500, 1000)

**Globalisation indices**, including:
- Political Globalisation Index (KOFPoGI)
- Economic Globalisation Index (KOFEcGI)
- Social Globalisation Index (KOFSoGI)

#### Why do we use unsupervised learning algorithm?
- Unsupervised learning algorithm is used to group similar data points into clusters without any predefined labels or categories.
- In this project, it helps uncover natural groupings of countries that show similar patterns in education systems and level of globalisation indices.
- This allows us to explore and compare international education trends in a data-driven way.

### Data Dictionary
The following features/variables are used in this analysis:

| Variable Name            | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `country_x`              | Country name                                     |
| `InboundRatio`           | Inbound mobility rate for international students (% of tertiary enrolment) |
| `InternationalStudentsNO`| Total number of international students enrolled                             |
| `KOFPoGI`                | Political Globalisation Index (from KOF Globalisation Index)                |
| `KOFEcGI`                | Economic Globalisation Index                                                |
| `KOFSoGI`                | Social Globalisation Index                                                  |
| `ISCED5 Percentage`      | Participation in short-cycle tertiary education (ISCED Level 5)             |
| `ISCED6 Percentage`      | Participation in Bachelor's level tertiary education (ISCED Level 6)        |
| `ISCED7 Percentage`      | Participation in Master's level tertiary education (ISCED Level 7)          |
| `ISCED8 Percentage`      | Participation in Doctoral level tertiary education (ISCED Level 8)          |
| `top 50 count`           | Number of universities ranked in the QS Top 50                              |
| `top 100 count`          | Number of universities ranked in the QS Top 100                             |
| `top 500 count`          | Number of universities ranked in the QS Top 500                             |
| `top 1000 count`         | Number of universities ranked in the QS Top 1000                            |
| `WESP`                   | World Economic Situation and Prospects 2021 grouping or economic tier       |

**Note:**  
Other decomposed KOF indicators (e.g. KOFGIdf, KOFTrGIdj, etc.) exist in the full dataset but are not used in this study for simplicity and relevance. This project focuses on understanding how globalisation and higher education metrics relate to inbound student mobility and university rankings.

### Data Cleaning and Preparation
- Checked for missing values across selected features/variables
- Selected features/variables based on the defined data dictionary.
- Removed rows with missing values to ensure data integrity for clustering
- Standardised all numerical features using StandardScaler to ensure equal contribution during clustering.

### Exploratory Data Analysis (EDA)

To understand the data structure and identify patterns before clustering, we performed:

**Summary statistics:**  
• Reviewed distributions, ranges, and central tendencies for all numerical features.

<img width="1032" height="409" alt="Screenshot 2025-07-19 at 22 44 22" src="https://github.com/user-attachments/assets/7dc4d0e0-776d-41e0-8773-d6bb68c41db4" />

**Visualisation - Percentage Distribution of Education Levels by Country:**

- Compared participation rates in ISCED Levels 5 to 8 across countries.  
- Used a vertical stacked bar chart to show the distribution of tertiary education participation by level.


<img width="468" height="14" alt="image" src="https://github.com/user-attachments/assets/492bf821-709b-469e-a076-09f427a52076" />


- This visual below shows that some countries have a more balanced distribution across all ISCED levels, while others are heavily concentrated in certain tiers (e.g., Bachelor’s or Master’s level).
- Australia and Sweden, both classified as developed countries (WESP) show high overall tertiary participation, with significant contributions across multiple ISCED levels, indicating a more developed and layered higher education system.


<img width="1490" height="590" alt="output_9_0" src="https://github.com/user-attachments/assets/0a46fba4-0544-4245-93fa-eecc3faeb075" />

** Visulisation - Percentage Distribution of University Rankings by Country:**  



<img width="987" height="1189" alt="output_10_0" src="https://github.com/user-attachments/assets/73229a78-f7e7-438d-955b-0bb234024aa2" />

