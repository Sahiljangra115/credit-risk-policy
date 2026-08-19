-- Prior Home Credit applications. Refusals here are the only observable rejects in the data.
SELECT
    SK_ID_CURR,
    COUNT(*)                                              AS prev_n,
    SUM((NAME_CONTRACT_STATUS = 'Approved')::INT)         AS prev_approved_n,
    SUM((NAME_CONTRACT_STATUS = 'Refused')::INT)          AS prev_refused_n,
    AVG((NAME_CONTRACT_STATUS = 'Refused')::INT)          AS prev_refused_rate,
    AVG(AMT_APPLICATION)                                  AS prev_amt_application_mean,
    AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))          AS prev_credit_app_ratio,
    MAX(DAYS_DECISION)                                    AS prev_last_decision_days
FROM previous_application
GROUP BY SK_ID_CURR;
