-- Application-level features. One row per SK_ID_CURR.
SELECT
    SK_ID_CURR,
    AMT_CREDIT / NULLIF(AMT_INCOME_TOTAL, 0)        AS credit_income_ratio,
    AMT_ANNUITY / NULLIF(AMT_INCOME_TOTAL, 0)       AS annuity_income_ratio,
    AMT_ANNUITY / NULLIF(AMT_CREDIT, 0)             AS annuity_credit_ratio,
    AMT_CREDIT / NULLIF(AMT_GOODS_PRICE, 0)         AS credit_goods_ratio,
    -DAYS_BIRTH / 365.25                            AS age_years,
    CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE -DAYS_EMPLOYED / 365.25 END AS employed_years,
    (CASE WHEN DAYS_EMPLOYED = 365243 THEN 1 ELSE 0 END) AS employed_flag_missing,
    (COALESCE(EXT_SOURCE_1, 0) + COALESCE(EXT_SOURCE_2, 0) + COALESCE(EXT_SOURCE_3, 0))
        / NULLIF((EXT_SOURCE_1 IS NOT NULL)::INT + (EXT_SOURCE_2 IS NOT NULL)::INT + (EXT_SOURCE_3 IS NOT NULL)::INT, 0)
                                                    AS ext_source_mean
FROM application;
