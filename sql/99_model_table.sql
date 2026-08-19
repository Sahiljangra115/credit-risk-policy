-- Final modelling table. Left joins keep every application; missing history stays NULL (informative).
SELECT
    a.SK_ID_CURR, a.TARGET, a.CODE_GENDER, a.REGION_RATING_CLIENT, a.app_month,
    a.AMT_CREDIT, a.AMT_INCOME_TOTAL, a.AMT_ANNUITY, a.CNT_CHILDREN, a.CNT_FAM_MEMBERS,
    a.NAME_CONTRACT_TYPE, a.NAME_INCOME_TYPE, a.NAME_EDUCATION_TYPE, a.NAME_FAMILY_STATUS,
    a.NAME_HOUSING_TYPE, a.FLAG_OWN_CAR, a.FLAG_OWN_REALTY, a.OCCUPATION_TYPE,
    a.EXT_SOURCE_1, a.EXT_SOURCE_2, a.EXT_SOURCE_3,
    f1.* EXCLUDE (SK_ID_CURR),
    f2.* EXCLUDE (SK_ID_CURR),
    f3.* EXCLUDE (SK_ID_CURR),
    f4.* EXCLUDE (SK_ID_CURR),
    f5.* EXCLUDE (SK_ID_CURR),
    f6.* EXCLUDE (SK_ID_CURR)
FROM app_base a
LEFT JOIN feat_application f1 USING (SK_ID_CURR)
LEFT JOIN feat_bureau       f2 USING (SK_ID_CURR)
LEFT JOIN feat_prev         f3 USING (SK_ID_CURR)
LEFT JOIN feat_installments f4 USING (SK_ID_CURR)
LEFT JOIN feat_pos          f5 USING (SK_ID_CURR)
LEFT JOIN feat_cc           f6 USING (SK_ID_CURR);
