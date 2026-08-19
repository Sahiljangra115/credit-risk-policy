SELECT
    SK_ID_CURR,
    COUNT(*)                                                        AS cc_n,
    AVG(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))           AS cc_utilisation_mean,
    MAX(AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0))           AS cc_utilisation_max,
    AVG(AMT_DRAWINGS_CURRENT)                                       AS cc_drawings_mean,
    MAX(SK_DPD)                                                     AS cc_dpd_max,
    AVG((SK_DPD > 0)::INT)                                          AS cc_dpd_share
FROM credit_card_balance
GROUP BY SK_ID_CURR;
