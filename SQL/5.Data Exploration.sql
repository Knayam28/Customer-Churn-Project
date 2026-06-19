SELECT DISTINCT PreferredPaymentMode FROM ecommerce_data;

SELECT COUNT(DISTINCT PreferredPaymentMode) AS Unique_Categories
FROM ecommerce_data;

SELECT
    PreferredPaymentMode,
    COUNT(*) AS Count
FROM ecommerce_data
GROUP BY PreferredPaymentMode
ORDER BY Count DESC;


SELECT DISTINCT PreferredLoginDevice FROM ecommerce_data;
SELECT DISTINCT Gender FROM ecommerce_data;
SELECT DISTINCT PreferedOrderCat FROM ecommerce_data;
SELECT DISTINCT MaritalStatus FROM ecommerce_data;
SELECT DISTINCT PreferredPaymentMode FROM ecommerce_data;
