WITH rfm_scores AS (
    SELECT
        CustomerID,
        Churn,
        5 - NTILE(4) OVER (ORDER BY DaySinceLastOrder) AS R_Score,
        NTILE(4) OVER (ORDER BY OrderCount) AS F_Score,
        NTILE(4) OVER (ORDER BY CashbackAmount) AS M_Score
    FROM ecommerce_data
    WHERE DaySinceLastOrder IS NOT NULL
      AND OrderCount IS NOT NULL
      AND CashbackAmount IS NOT NULL
),

rfm_segments AS (
    SELECT *,
           (R_Score + F_Score + M_Score) AS RFM_Total
    FROM rfm_scores
)

SELECT
    CASE
        WHEN RFM_Total >= 10 THEN 'VIP'
        WHEN RFM_Total >= 7 THEN 'Loyal'
        WHEN RFM_Total >= 5 THEN 'At Risk'
        ELSE 'Lost'
    END AS Customer_Segment,

    COUNT(*) AS Customers,
	SUM(Churn) AS Churned_Customers,
    ROUND(AVG(Churn)*100,2) AS Churn_Rate 

FROM rfm_segments
GROUP BY Customer_Segment
ORDER BY Churn_Rate DESC;

WITH rfm_scores AS (
    SELECT
        CustomerID,

        5 - NTILE(4) OVER (ORDER BY DaySinceLastOrder) AS R_Score,

        NTILE(4) OVER (ORDER BY OrderCount) AS F_Score,

        NTILE(4) OVER (ORDER BY CashbackAmount) AS M_Score

    FROM ecommerce_data
    WHERE DaySinceLastOrder IS NOT NULL
      AND OrderCount IS NOT NULL
      AND CashbackAmount IS NOT NULL
),

rfm_segments AS (
    SELECT
        CustomerID,
        R_Score,
        F_Score,
        M_Score,
        (R_Score + F_Score + M_Score) AS RFM_Total,

        CASE
            WHEN (R_Score + F_Score + M_Score) >= 10 THEN 'VIP'
            WHEN (R_Score + F_Score + M_Score) >= 7 THEN 'Loyal'
            WHEN (R_Score + F_Score + M_Score) >= 5 THEN 'At Risk'
            ELSE 'Lost'
        END AS Customer_Segment

    FROM rfm_scores
)

SELECT *
FROM rfm_segments;