# **Dataset Overview**

This dataset captures detailed information about customer behavior and purchasing patterns, primarily from an e-commerce or retail platform. It includes various features that help in analyzing how customers interact with the site, their purchasing habits, and some demographic details. This is a synthetic dataset generated for educational purposes by Gretel AI.

## **Key Variables and Descriptions**

### **Customer ID**
A unique identifier assigned to each customer. This attribute is essential for tracking individual customer behavior across different features in the dataset.

### **Age**
Records the age of each customer. This attribute is useful for demographic analysis, helping to identify age-related trends and preferences in purchasing behavior.

### **Gender**
Indicates the gender of each customer (e.g., Male, Female). Analyzing this attribute allows for understanding gender-specific trends in product preferences and purchasing habits.

### **Location**
Represents the geographic location of the customer. This attribute helps in identifying regional trends and differences in customer behavior, which can inform location-based marketing strategies.

### **Annual Income**
Reflects the annual income of the customer. This financial attribute is critical for segmenting customers by their purchasing power and understanding the relationship between income levels and buying patterns.

### **Purchase History**
A list of products that each customer has purchased, including details such as the purchase date, product category, and price. It provides insights into customer preferences and helps in analyzing repeat purchases and category-specific trends.

### **Browsing History**
A log of products viewed by each customer, along with timestamps. This attribute is important for understanding customer interest and engagement, even if a purchase was not made.

### **Product Reviews**
Contains text reviews written by customers about products they have purchased, along with a rating (1-5 stars). This attribute provides qualitative insights into customer satisfaction and product quality.

### **Time on Site**
Measures the total time a customer spends on the platform during their visit. It is a key metric for analyzing user engagement and can be correlated with browsing and purchasing behavior.

## **Data Summary**

- **Customer ID**
  - **Range:** 1001.00 - 1013.00
  - **Mean:** 1007.00
  - **Standard Deviation:** 3.59

- **Age**
  - **Range:** 24.00 - 65.00
  - **Mean:** 40
  - **Standard Deviation:** 11

- **Gender**
  - **Distribution:** Female (52%), Male (36%), Other (12%)

- **Location**
  - **Most Common Locations:** City D (24%), City E (12%), Other (64%)

- **Annual Income**
  - **Range:** 40,000.00 - 100,000.00
  - **Mean:** 65,800.00
  - **Standard Deviation:** 16,900.00

- **Purchase History**
  - **Example Entries:** 
    - `[{"Date": "2022-03-05", "Category": "Clothing", "Price": 34.99}, ...]`
  
- **Browsing History**
  - **Example Entries:** 
    - `[{"Timestamp": "2022-03-10T14:30:00Z"}, ...]`

- **Product Reviews**
  - **Example Reviews:** 
    - `{"Review Text": "Excellent product, highly recommend!", "Rating": 5}`

- **Time on Site**
  - **Range:** 32.50 - 486.30 minutes
  - **Mean:** 233 minutes
  - **Standard Deviation:** 109 minutes

## **Collection Methodology**

The dataset was generated using machine learning algorithms that simulate typical customer interactions with an e-commerce platform. The methodology involves:
- **Pattern Recognition:** Identifying and reproducing patterns seen in real-world customer data.
- **Synthetic Data Generation:** Creating data points for each feature based on recognized patterns.
- **Controlled Variation:** Incorporating controlled variations to ensure diversity while maintaining realistic relationships.

## **License**

This dataset is licensed under the [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license.

## **Kaggle Link**

You can access the dataset on Kaggle [here](https://www.kaggle.com/datasets/paulsamuelwe/e-commerce-customer-behaviour-dataset).
