-- Bureau + bureau_balance. bureau_balance is rolled up per bureau loan first, then per customer.
WITH bb AS (
    SELECT
        SK_ID_BUREAU,
        COUNT(*)                                                     AS bb_months,
        AVG((STATUS IN ('1','2','3','4','5'))::INT)                  AS bb_late_share,
        MAX(CASE WHEN STATUS IN ('1','2','3','4','5') THEN STATUS::INT ELSE 0 END) AS bb_worst_status
    FROM bureau_balance
    GROUP BY SK_ID_BUREAU
)
SELECT
    b.SK_ID_CURR,
    COUNT(*)                                              AS bureau_n,
    SUM((b.CREDIT_ACTIVE = 'Active')::INT)                AS bureau_active_n,
    COUNT(DISTINCT b.CREDIT_TYPE)                         AS bureau_type_n,
    SUM(b.AMT_CREDIT_SUM_DEBT)                            AS bureau_debt_sum,
    SUM(b.AMT_CREDIT_SUM_DEBT) / NULLIF(SUM(b.AMT_CREDIT_SUM), 0) AS bureau_debt_ratio,
    SUM(b.AMT_CREDIT_SUM_OVERDUE)                         AS bureau_overdue_sum,
    MAX(b.CREDIT_DAY_OVERDUE)                             AS bureau_max_day_overdue,
    AVG(bb.bb_late_share)                                 AS bureau_bb_late_share,
    MAX(bb.bb_worst_status)                               AS bureau_bb_worst_status
FROM bureau b
LEFT JOIN bb USING (SK_ID_BUREAU)
GROUP BY b.SK_ID_CURR;
