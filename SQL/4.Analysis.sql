SELECT * FROM churn.ecommerce_data;

# Total Customers
SELECT COUNT(*) AS Total_Rows 
FROM ecommerce_data;

# Total Columns
SELECT COUNT(*) AS Total_Columns
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'ecommerce_data'
-- Assuming you are actively using your database, DATABASE() grabs the current one
AND TABLE_SCHEMA = DATABASE();

# Overall Churn Rate
SELECT ROUND(AVG(Churn)*100,2) AS Churn_Rate_Percent
FROM ecommerce_data;

# Check Class Balance
SELECT Churn, COUNT(*) AS Customers
FROM ecommerce_data
GROUP BY Churn;

-- Churn by Gender
SELECT Gender,
COUNT(*) AS Customers,
SUM(Churn) AS Churned_Customers,
ROUND(AVG(Churn) * 100, 2) AS Churn_Rate
FROM ecommerce_data
GROUP BY Gender;

# Churn by City Tier
SELECT CityTier,
COUNT(*) AS Customers,
SUM(Churn) AS Churned_Customers,
ROUND(AVG(Churn)*100,2) AS Churn_Rate
FROM ecommerce_data
GROUP BY CityTier
ORDER BY CityTier;

-- Complaint Impact on Churn
SELECT Complain,
COUNT(*) AS Customers,
SUM(Churn) AS Churned_Customers,
ROUND(AVG(Churn)*100,2) AS Churn_Rate
FROM ecommerce_data
GROUP BY Complain;

-- Churn by Tenure Group 
SELECT
   CASE
    WHEN Tenure BETWEEN 0 AND 3 THEN '0-3 Months'
    WHEN Tenure BETWEEN 4 AND 6 THEN '4-6 Months'
    WHEN Tenure BETWEEN 7 AND 12 THEN '7-12 Months'
    WHEN Tenure BETWEEN 13 AND 24 THEN '13-24 Months'
    ELSE '25+ Months'
    END AS Tenure_Group,
    COUNT(*) AS Customers,
    SUM(Churn) AS Churned_Customers,
    ROUND(AVG(Churn)*100,2) AS Churn_Rate
FROM ecommerce_data
WHERE Tenure IS NOT NULL
GROUP BY Tenure_Group
ORDER BY Churn_Rate desc;

-- SELECT
--     COUNT(*) AS Customers,
--     SUM(Churn) AS Churned,
--     ROUND(AVG(Churn)*100,2) AS Churn_Rate
-- FROM ecommerce_data
-- WHERE Tenure >= 25;

-- Missing Value Summary
SELECT
    SUM(Tenure IS NULL) AS Missing_Tenure,
    SUM(WarehouseToHome IS NULL) AS Missing_Warehouse,
    SUM(HourSpendOnApp IS NULL) AS Missing_AppHours,
    SUM(OrderAmountHikeFromlastYear IS NULL) AS Missing_OrderHike,
    SUM(CouponUsed IS NULL) AS Missing_CouponUsed,
    SUM(OrderCount IS NULL) AS Missing_OrderCount,
    SUM(DaySinceLastOrder IS NULL) AS Missing_LastOrder
FROM ecommerce_data;