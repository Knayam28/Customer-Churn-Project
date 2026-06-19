SET GLOBAL local_infile = 1;
SET autocommit = 0;
SET foreign_key_checks = 0;
SET unique_checks = 0;

LOAD DATA LOCAL INFILE 'C:/Users/mohit/Desktop/ecommerce_churn_project/E_Commerce_Dataset.csv'
INTO TABLE ecommerce_data
FIELDS TERMINATED BY ',' 
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS

-- 1. Read EVERY incoming column into a temporary @variable
(@CustomerID, @Churn, @Tenure, @PreferredLoginDevice, @CityTier, @WarehouseToHome, @PreferredPaymentMode, @Gender, @HourSpendOnApp, @NumberOfDeviceRegistered, @PreferedOrderCat, @SatisfactionScore, @MaritalStatus, @NumberOfAddress, @Complain, @OrderAmountHikeFromlastYear, @CouponUsed, @OrderCount, @DaySinceLastOrder, @CashbackAmount)

-- 2. Map every variable to the actual column, converting blanks to NULL
SET 
    CustomerID = NULLIF(@CustomerID, ''),
    Churn = NULLIF(@Churn, ''),
    Tenure = NULLIF(@Tenure, ''),
    PreferredLoginDevice = NULLIF(@PreferredLoginDevice, ''),
    CityTier = NULLIF(@CityTier, ''),
    WarehouseToHome = NULLIF(@WarehouseToHome, ''),
    PreferredPaymentMode = NULLIF(@PreferredPaymentMode, ''),
    Gender = NULLIF(@Gender, ''),
    HourSpendOnApp = NULLIF(@HourSpendOnApp, ''),
    NumberOfDeviceRegistered = NULLIF(@NumberOfDeviceRegistered, ''),
    PreferedOrderCat = NULLIF(@PreferedOrderCat, ''),
    SatisfactionScore = NULLIF(@SatisfactionScore, ''),
    MaritalStatus = NULLIF(@MaritalStatus, ''),
    NumberOfAddress = NULLIF(@NumberOfAddress, ''),
    Complain = NULLIF(@Complain, ''),
    OrderAmountHikeFromlastYear = NULLIF(@OrderAmountHikeFromlastYear, ''),
    CouponUsed = NULLIF(@CouponUsed, ''),
    OrderCount = NULLIF(@OrderCount, ''),
    DaySinceLastOrder = NULLIF(@DaySinceLastOrder, ''),
    CashbackAmount = NULLIF(@CashbackAmount, '');

COMMIT;

SET foreign_key_checks = 1;
SET unique_checks = 1;
SET autocommit = 1;