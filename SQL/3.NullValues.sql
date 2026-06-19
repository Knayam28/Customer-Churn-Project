SELECT 
    SUM(CASE WHEN CustomerID IS NULL THEN 1 ELSE 0 END) AS CustomerID_Nulls,
    SUM(CASE WHEN Churn IS NULL THEN 1 ELSE 0 END) AS Churn_Nulls,
    SUM(CASE WHEN Tenure IS NULL THEN 1 ELSE 0 END) AS Tenure_Nulls,
    SUM(CASE WHEN PreferredLoginDevice IS NULL OR PreferredLoginDevice = '' THEN 1 ELSE 0 END) AS LoginDevice_Nulls,
    SUM(CASE WHEN CityTier IS NULL THEN 1 ELSE 0 END) AS CityTier_Nulls,
    SUM(CASE WHEN WarehouseToHome IS NULL THEN 1 ELSE 0 END) AS WarehouseToHome_Nulls,
    SUM(CASE WHEN PreferredPaymentMode IS NULL OR PreferredPaymentMode = '' THEN 1 ELSE 0 END) AS PaymentMode_Nulls,
    SUM(CASE WHEN Gender IS NULL OR Gender = '' THEN 1 ELSE 0 END) AS Gender_Nulls,
    SUM(CASE WHEN HourSpendOnApp IS NULL THEN 1 ELSE 0 END) AS HourSpendOnApp_Nulls,
    SUM(CASE WHEN NumberOfDeviceRegistered IS NULL THEN 1 ELSE 0 END) AS Devices_Nulls,
    SUM(CASE WHEN PreferedOrderCat IS NULL OR PreferedOrderCat = '' THEN 1 ELSE 0 END) AS OrderCat_Nulls,
    SUM(CASE WHEN SatisfactionScore IS NULL THEN 1 ELSE 0 END) AS Satisfaction_Nulls,
    SUM(CASE WHEN MaritalStatus IS NULL OR MaritalStatus = '' THEN 1 ELSE 0 END) AS MaritalStatus_Nulls,
    SUM(CASE WHEN NumberOfAddress IS NULL THEN 1 ELSE 0 END) AS NumAddress_Nulls,
    SUM(CASE WHEN Complain IS NULL THEN 1 ELSE 0 END) AS Complain_Nulls,
    SUM(CASE WHEN OrderAmountHikeFromlastYear IS NULL THEN 1 ELSE 0 END) AS OrderHike_Nulls,
    SUM(CASE WHEN CouponUsed IS NULL THEN 1 ELSE 0 END) AS CouponUsed_Nulls,
    SUM(CASE WHEN OrderCount IS NULL THEN 1 ELSE 0 END) AS OrderCount_Nulls,
    SUM(CASE WHEN DaySinceLastOrder IS NULL THEN 1 ELSE 0 END) AS DaySinceLastOrder_Nulls,
    SUM(CASE WHEN CashbackAmount IS NULL THEN 1 ELSE 0 END) AS Cashback_Nulls
FROM ecommerce_data;

