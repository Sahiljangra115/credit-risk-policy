SELECT
    SK_ID_CURR,
    COUNT(*)                                        AS pos_n,
    AVG((SK_DPD > 0)::INT)                          AS pos_dpd_share,
    MAX(SK_DPD)                                     AS pos_dpd_max,
    SUM((NAME_CONTRACT_STATUS = 'Active')::INT)     AS pos_active_rows,
    AVG(CNT_INSTALMENT)                             AS pos_cnt_instalment_mean
FROM pos_cash_balance
GROUP BY SK_ID_CURR;
