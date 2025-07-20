## Global Higher Education Clustering Analysis with Python: Unsupervised Learning Algorithm

### 1. Project Overview
This project aims to understand how international student mobility is related to the level of globalisation and the structure of higher education systems across countries in 2019.
We apply unsupervised learning, specifically clustering analysis, to group countries based on similarities in:

**Education system characteristics**, including:  
- ISCED Participation Levels (Levels 5 to 8)
- QS World University Rankings (Top 50, 100, 500 and 1000)

**Globalisation indices**, including:
- Political Globalisation Index (KOFPoGI)
- Economic Globalisation Index (KOFEcGI)
- Social Globalisation Index (KOFSoGI)

#### Why do we use unsupervised learning algorithm?
- An unsupervised learning algorithm groups similar data points into clusters without any predefined labels or categories.
- In this analysis, it helps uncover natural groupings of countries that share similar patterns in education systems and level of globalisation indices.

### 2. Data Dictionary
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
Other decomposed KOF indicators (e.g. KOFGIdf, KOFTrGIdj, etc.) exist in the full dataset but are not used in this study for simplicity and relevance. This analysis focuses on understanding how globalisation and higher education metrics relate to inbound student mobility and university rankings.

### 3. Data Cleaning and Preparation
- Checked for missing values across selected features/variables
- Selected features/variables based on the defined data dictionary.
- Removed rows with missing values to ensure data integrity for clustering
- Standardised all numerical features to ensure equal contribution during clustering analysis.

### 4. Exploratory Data Analysis (EDA)

To understand the data structure and identify patterns before applying clustering analysis, we conducted:

**4.1 Summary statistics**  

Across the 42 countries studied, several key patterns were observed in relation to international student engagement:

- On average, 9% of all students enrolled in tertiary education were international students, though this varied widely — from as low as 0.22% to as high as 35.29%, depending on the country.
- The number of international students also differed significantly: the average was 117,317 students, but some countries hosted nearly one million (maximum = 976,562), while others had only a few thousand (minimum = 1,546).
- Bachelor’s level (ISCED 6) dominated tertiary education systems, with an average share of 45.24%, while master’s level (ISCED 7) follows at 14.23%.
- Doctoral level tertiary education participation (ISCED 8) was the smallest segment, averaging just 2.10%. Most countries fell between 0.00% and 5.15%, suggesting this level remained specialised and selective due to longer study duration, competitive entry and limited institutional capacity or funding.
- Overall globalisation indices were relatively high across the sample, with political = 84.95, economic = 71.98 and social = 79.98 on average.
<img width="790" height="408" alt="Screenshot 2025-07-20 at 13 54 10" src="https://github.com/user-attachments/assets/862a6d1f-e139-40d7-9038-2d353d5b6c34" />

**4.2 Visualisation - Percentage Distribution of Education Levels by Country:**

- Participation rates in ISCED Levels 5 to 8 were compared across countries using a vertical stacked bar chart to show the distribution of tertiary education participation by level.
- The visual below indicated that some countries had a more balanced distribution across all ISCED levels, while others were heavily concentrated in certain tiers (e.g. bachelor’s or master’s level).
- Australia and Sweden, both classified as developed countries (WESP), showed high overall tertiary participation with significant contributions across multiple ISCED levels, indicating a more developed and layered higher education system..
- Turkey showed relatively even participation across short-cycle tertiary (e.g. associate degree or foundation degree), bachelor’s and master’s levels (ISCED Levels 5, 6 and 7 respectively), but had comparatively limited engagement at the doctoral level (ISCED 8).

<img width="1490" height="590" alt="Percentage_Education_Levels" src="https://github.com/user-attachments/assets/7c397215-69a3-469c-9a0f-1682ee630a27" />


**4.3 Visulisation - Percentage Distribution of University Rankings by Country:**  
- The percentage distribution of universities ranked in the top 50, top 100, top 500 and top 1000 was compared across countries using a horizontal stacked bar chart.
- - The visual showed that some countries (e.g. Jordan, Latvia, Slovakia, Romania and Slovenia) had all of their ranked universities only in the top 1000 tier, highlighting limited presence in tiers with higher rankings.
- In contrast, countries like Ireland, Finland and Germany demonstrated a more even spread of university representation across the top 100 and top 500 tiers, suggesting broader academic strength.
-  New Zealand showed a strong mid-tier presence, with nearly half of its ranked universities falling in the top 500 tier and the rest in the top 1000. Only 6% of its universities appeared in the top 50.
- Countries such as the United States, United Kingdom, Switzerland and Australia showed stronger representation in the top 50 and top 100, with many of their institutions recognised as leading at the global level.
<img width="987" height="1189" alt="Percentage_University_Ranking" src="https://github.com/user-attachments/assets/63b08572-9743-4876-87a8-9fa5d48f0463" />

### 5. Data Analysis – Clustering Analysis

This section used clustering analysis to explore how countries naturally group based on similarities in tertiary education participation and globalisation indices. No predefined categories were applied as the aim was to let the data speak and reveal which countries inclined to follow similar patterns.


**5.1  Identifying Optimal Clusters Using K-Means Clustering (Elbow Curve)**

- Before applying clustering analysis, we excluded the country_x and WESP variables as they are non-numeric and not suitable for clustering algorithms that require numerical input.
- All remaining variables were standardised to ensure fair comparison across different scales.
- We then needed to decide how many meaningful groups (clusters) to divide the countries into.
- The “Elbow Curve” method was used to determine the optimal number of clusters. The elbow point is where the curve starts to flatten, indicating that adding more clusters beyond this point provides only marginal improvement in reducing within-cluster variance.
- We plotted k-values from 1 to 10 and observed that the elbow occurred at k = 3. This suggested that dividing the countries into three clusters captured key patterns without overfitting.
- This method aligned with the purpose of using unsupervised learning, as it allowed natural groupings to emerge from the data without relying on predefined categories.

<img width="989" height="490" alt="Elbow_Curve" src="https://github.com/user-attachments/assets/b5c2cb83-7ad9-4914-b8eb-e58b8ca23f45" />


**5.2 Exploring Country Groupings Using Hierarchical Clustering (Dendrogram)** 

- After identifying that three clusters were optimal using the Elbow Method, we further explored the natural structure of the data using Hierarchical Clustering. This method
  produces a dendrogram, creating a tree-like structure that shows how countries are grouped based on similarity across the following indicators:
  - International student mobility,
  - Tertiary education participation (ISCED 5–8),
  - QS World University Rankings (Top 50, 100, 500, 1000),
  - Inbound ratio and number of international students, and
  - Globalisation indices (Political, Economic and Social Globalisation Index).

- Key Insights:
  - Countries that merged at shorter horizontal distances were more similar to each other in their overall profiles across all indicators
  - The color bands (e.g., green, blue, orange) represented how countries were grouped together at different levels of similarity. They helped illustrate the hierarchy of clusters
    formed during the clustering anlaysis process.

<img width="790" height="690" alt="ierarchical _Clustering_Dendrogram)" src="https://github.com/user-attachments/assets/51f5bfe4-40e4-447d-a649-608b2edf7a2e" />


**5.3 Exploring Cluster Analysis Between Master’s-Level Participation and Economic Globalisation Index** 

- To further interpret the cluster profiles identified in Section 5.2, we examined how they relate to Master’s-level participation (ISCED 7) and the Economic Globalisation Index (KOFEcGI). This two-dimensional scatter plot provides a visual comparison of how countries vary in terms of postgraduate education structure and global economic integration.
- In this analysis, these two variables were selected to explore potential relationships between economic globalisation and a country’s capacity to support international postgraduate students.

Key insights:

  - 🟢 Cluster 0 (green) included countries such as Germany, Netherlands, New Zealand, Sweden, Austria, Australia and Ireland. These countries recorded both high Master’s-level participation and strong economic globalisation scores. This reflects well-developed postgraduate systems and high levels of global integration, making them attractive destinations for international students.
  - 🟠 Cluster 1 (orange) included countries like the United States and United Kingdom. While these countries were highly globalised economically, their Master’s-level participation was comparatively moderate. Although globally connected, the proportion of students in Master’s programmes was lower than in countries in Cluster 0.
  - 🔴 Cluster 2 (red) consisted of countries such as China, Brazil, Argentina and South Africa. These countries had lower values for both Master’s-level participation and economic globalisation. This pattern suggests more limited engagement with international postgraduate education, potentially due to domestic education priorities or barriers to cross-border academic exchange.

<img width="1975" height="790" alt="output_14_0" src="https://github.com/user-attachments/assets/781e1a1c-e528-422b-b273-dfd290207755" />


**Conclusion**

The insights in this analysis highlighted and demonstrated the value of a data-driven approach using unsupervised learning clustering to examine how combining globalisation indicators and education system characteristics can uncover meaningful natural patterns in international student mobility, without relying on predefined categories. Clustering analysis enabled natural groupings to emerge from the data, showing how countries varied in their ability to attract and support international students through structural and global factors.

**Project Files**

*- Python Code*:
  [View Python_Code.ipynb](https://github.com/zar-moethu/world-education-systems-python/blob/main/Python_Code.ipynb)

*- Dataset*:
[Download raw file](https://github.com/zar-moethu/world-education-systems-python/raw/main/dataset.xlsx)

*- Output Visuals*:
[Output folder](https://github.com/zar-moethu/world-education-systems-python/tree/main/Output).


**📘How to Use This**

- Data analysis is an ongoing learning journey, focused on exploring data meaningfully rather than producing a finalised or commercial product.
- This work is a living document, open to iteration, feedback and collaboration, as its value lies not only in the results but also in the learning that comes from the process of discovery.
- Feel free to download the code, dataset and explore the notebooks or visualisations for your own learning or inspiration.

⚠️ Disclaimer: This dashboard is intended for visual learning and demonstration purposes only — not for business or operational use.
