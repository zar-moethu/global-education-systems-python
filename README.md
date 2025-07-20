## Global Higher Education Clustering Analysis with Python: Unsupervised Learning Algorithm

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
| `country_x`              | Country name                                                                |
| `InboundRatio`           | Inbound mobility rate for international students                            |
| `InternationalStudentsNO`| Total number of inbound international students                              |
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
| `WESP`                   | World Economic Situation and Prospects|

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

Across the 42 countries studied, several key patterns were observed in relation to international student engagement:
- On average, 9% of all students enrolled in tertiary education are international students, though this varies widely — from as low as 0.22% to as high as 35.29%, depending on the country. The number of international students also differs significantly: while the average is 117,317 students, some countries host nearly one million (maximum = 976,562), while others have only a few thousand.
- Bachelor’s level (ISCED 6) dominates tertiary education systems, with an average share of 45.24%, while master’s level (ISCED 7) follows at 14.23%.
- Doctoral level tertiary education participation (ISCED 8) is the smallest segment, averaging just 2.10%. Most countries fall between 0.00% and 5.15%, suggesting this level remains specialised and selective due to longer study duration, competitive entry and limited institutional capacity or funding.
- Overall globalisation indices are relatively high across the sample, with political = 84.95, economic = 71.98 and social = 79.98 on average.
<img width="790" height="408" alt="Screenshot 2025-07-20 at 13 54 10" src="https://github.com/user-attachments/assets/862a6d1f-e139-40d7-9038-2d353d5b6c34" />

**Visualisation - Percentage Distribution of Education Levels by Country:**

- Compared participation rates in ISCED Levels 5 to 8 across countries using a vertical stacked bar chart to show the distribution of tertiary education participation by level.
- The visual below indicated that some countries had a more balanced distribution across all ISCED levels, while others were heavily concentrated in certain tiers (e.g. bachelor’s or master’s level).
- Australia and Sweden, both classified as developed countries (WESP), showed high overall tertiary participation, with significant contributions across multiple ISCED levels, indicating a more developed and layered higher education system.
- Turkey showed relatively even participation across short-cycle tertiary (e.g. associate degree or foundation degree), bachelor’s and master’s levels (ISCED Levels 5, 6 and 7 respectively), but comparatively limited engagement at the doctoral level (ISCED 8)..

<img width="1490" height="590" alt="output_9_0" src="https://github.com/user-attachments/assets/92d04227-1ccc-45f5-b310-6a26a9467fd5" />


**Visulisation - Percentage Distribution of University Rankings by Country:**  
- Compared the percentage distribution of universities ranked in the top 50, top 100, top 500 and top 1000 across countries using a horizontal stacked bar chart.
- The visual showed that some countries (e.g. Jordan, Latvia, Slovakia, Romania and Slovenia) had all of their ranked universities only in the top 1000 tier, highlighting limited presence in tiers with higher rankings.
- In contrast, countries like Ireland, Finland and Germany demonstrated a more even spread of university representation across the top 100 and top 500 tiers, suggesting broader academic strength.
- New Zealand showed a strong mid-tier presence, with nearly half of its ranked universities falling in the top 500 tier and the rest in the top 1000. Only 6% of its universities appeared in the top 50.
- Countries such as the United States, United Kingdom, Switzerland and Australia showed stronger representation in the top 50 and top 100, indicating a significant share of world-leading institutions.
  
<img width="987" height="1189" alt="output_10_0" src="https://github.com/user-attachments/assets/0ad405a8-899f-4439-a965-b68ef240443f" />

### Data Analysis – Discovering Patterns Using Unsupervised learning clustering
