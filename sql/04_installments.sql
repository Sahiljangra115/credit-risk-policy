-- Repayment behaviour. DAYS_* are negative days before the application.
SELECT
    SK_ID_CURR,
    COUNT(*)                                                                AS inst_n,
    AVG(GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0))                  AS inst_days_late_mean,
    MAX(GREATEST(DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT, 0))                  AS inst_days_late_max,
    AVG((DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT)::INT)                        AS inst_late_share,
    AVG(CASE WHEN DAYS_INSTALMENT >= -365
             THEN (DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT)::INT END)          AS inst_late_share_12m,
    AVG(AMT_PAYMENT / NULLIF(AMT_INSTALMENT, 0))                            AS inst_pay_ratio_mean,
    SUM((AMT_PAYMENT < AMT_INSTALMENT - 0.01)::INT)                         AS inst_underpaid_n
FROM installments_payments
GROUP BY SK_ID_CURR;
