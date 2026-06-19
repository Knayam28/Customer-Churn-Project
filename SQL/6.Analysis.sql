-- Churn by Order Category
SELECT
    CASE
        WHEN PreferedOrderCat IN ('Mobile','Mobile Phone')
            THEN 'Mobile'
        ELSE PreferedOrderCat
    END AS OrderCategory,
    COUNT(*) AS Customers,
    SUM(Churn) AS Churned_Customers,
    ROUND(AVG(Churn)*100,2) AS ChurnRate
FROM ecommerce_data
GROUP BY OrderCategory
ORDER BY ChurnRate DESC;

-- Churn by Payment Category
SELECT
    CASE
		WHEN PreferredPaymentMode IN ('COD','Cash on Delivery') THEN 'Cash on Delivery'
		WHEN PreferredPaymentMode IN ('CC','Credit Card') THEN 'Credit Card'
		WHEN PreferredPaymentMode = 'E wallet' THEN 'E Wallet'
		ELSE PreferredPaymentMode
    END as Paymentcategory,
    COUNT(*) AS Customers,
    SUM(Churn) AS Churned_Customers,
    ROUND(AVG(Churn)*100,2) AS ChurnRate
FROM ecommerce_data
GROUP BY Paymentcategory
ORDER BY ChurnRate DESC;

-- Churn by Login Device
SELECT
    CASE
        WHEN PreferredLoginDevice IN ('Phone','Mobile Phone') THEN 'Mobile'
        ELSE PreferredLoginDevice
    END AS LoginDevice,
    COUNT(*) AS Customers,
    SUM(Churn) AS Churned_Customers,
    ROUND(AVG(Churn)*100,2) AS ChurnRate
FROM ecommerce_data
GROUP BY LoginDevice
ORDER BY ChurnRate DESC;

-- Churn by Marital Status
SELECT
    MaritalStatus,
    COUNT(*) AS Customers,
    SUM(Churn) AS Churned_Customers,
    ROUND(AVG(Churn)*100,2) AS ChurnRate
FROM ecommerce_data
GROUP BY MaritalStatus
ORDER BY ChurnRate DESC;

-- Churn by Number of Addresses
SELECT
    NumberOfAddress,
    COUNT(*) AS Customers,
    ROUND(AVG(Churn)*100,2) AS ChurnRate
FROM ecommerce_data
GROUP BY NumberOfAddress
ORDER BY NumberOfAddress;

-- Complaint + Tenure
SELECT
    Complain,
    CASE
		WHEN Tenure BETWEEN 0 AND 3 THEN '0-3 Months'
		WHEN Tenure BETWEEN 4 AND 6 THEN '4-6 Months'
		WHEN Tenure BETWEEN 7 AND 12 THEN '7-12 Months'
		WHEN Tenure BETWEEN 13 AND 24 THEN '13-24 Months'
    ELSE '25+ Months'
    END AS TenureGroup,
    COUNT(*) AS Customers,
    ROUND(AVG(Churn)*100,2) AS ChurnRate
FROM ecommerce_data
WHERE Tenure IS NOT NULL
GROUP BY Complain, TenureGroup
ORDER BY Complain, ChurnRate DESC;

-- Cashback Segmentation
SELECT
    CASE
        WHEN CashbackAmount < 100 THEN '0-100'
        WHEN CashbackAmount < 200 THEN '100-200'
        ELSE '200+'
    END AS CashbackGroup,
    COUNT(*) AS Customers,
    ROUND(AVG(Churn)*100,2) AS ChurnRate
FROM ecommerce_data
GROUP BY CashbackGroup
ORDER BY ChurnRate DESC;

-- Mobile + Payment Mode
SELECT
    CASE
        WHEN PreferedOrderCat IN ('Mobile','Mobile Phone')
        THEN 'Mobile'
        ELSE PreferedOrderCat
    END AS OrderCategory,
    CASE
        WHEN PreferredPaymentMode IN ('COD','Cash on Delivery')
        THEN 'Cash on Delivery'
        WHEN PreferredPaymentMode IN ('CC','Credit Card')
        THEN 'Credit Card'
        ELSE PreferredPaymentMode
    END AS PaymentMode,
    COUNT(*) AS Customers,
    ROUND(AVG(Churn)*100,2) AS ChurnRate
FROM ecommerce_data
GROUP BY OrderCategory, PaymentMode
HAVING COUNT(*) > 30
ORDER BY ChurnRate DESC;



-- summary table:

-- Factor	High Risk Segment	Churn Rate
-- Tenure	0-3 Months + Complaint	66.09%
-- Marital Status	Single	26.73%
-- Order Category	Mobile	27.40%
-- Payment Mode	Cash on Delivery	24.90%
-- Cashback	100-200	18.90%
-- Product Category	Grocery	4.88%