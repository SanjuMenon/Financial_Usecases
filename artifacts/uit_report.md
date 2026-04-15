## Insider trading detection — model report (mock data)

### Model performance (holdout)
- **ROC AUC**: 0.6009

- **Class 0**: precision=0.864, recall=0.999, f1=0.927
- **Class 1**: precision=0.538, recall=0.009, f1=0.018
- **Accuracy**: 0.8636

### SHAP (global drivers)
Ranked by mean |SHAP| on the holdout set. Higher means the feature more strongly drives predictions.

| feature               |   mean_abs_shap |    mean_shap |
|-----------------------|-----------------|--------------|
| is_director           |      0.448687   | -0.0925673   |
| acq_disp_D            |      0.147658   | -0.0115883   |
| market_beta           |      0.125289   | -0.0126923   |
| prc_op_earnings_basic |      0.106493   | -0.0123649   |
| sprd_rtn              |      0.0972849  | -0.00497394  |
| ret                   |      0.0967089  | -0.00902374  |
| price_to_book         |      0.0942291  | -0.00954487  |
| hml_beta              |      0.0728621  | -0.00493279  |
| is_officer            |      0.0100325  | -0.000396418 |
| ten_percent_owner     |      0.00520309 | -9.74797e-05 |

### Causal forest (summary)
Mean marginal effects from the causal forest (mocked setup; interpret as direction/relative strength).

| treatment     |   mean_marginal_effect |
|---------------|------------------------|
| is_director   |             0.178992   |
| price_to_book |            -0.00355124 |
| market_beta   |            -0.0522712  |
| ret           |            -0.0322686  |

### Top flagged trades (examples)
Highest `uit_risk` rows with their top driver features.

|   trade_id |        cik |   personid | transaction_date    |   uit_risk | top_drivers                                                            |
|------------|------------|------------|---------------------|------------|------------------------------------------------------------------------|
|      24547 | 0001000066 |        765 | 2022-07-13 00:00:00 |   0.718647 | is_director,prc_op_earnings_basic,market_beta,sprd_rtn,price_to_book   |
|       4704 | 0001000225 |        966 | 2020-10-08 00:00:00 |   0.697564 | is_director,prc_op_earnings_basic,market_beta,hml_beta,sprd_rtn        |
|       5019 | 0001000142 |        845 | 2023-04-20 00:00:00 |   0.623076 | is_director,ret,prc_op_earnings_basic,acq_disp_D,price_to_book         |
|      16645 | 0001000186 |        392 | 2022-06-15 00:00:00 |   0.582285 | is_director,ret,price_to_book,acq_disp_D,prc_op_earnings_basic         |
|       7057 | 0001000242 |        853 | 2022-08-12 00:00:00 |   0.540432 | is_director,sprd_rtn,price_to_book,hml_beta,ret                        |
|       4806 | 0001000214 |        126 | 2023-05-17 00:00:00 |   0.538223 | is_director,price_to_book,ret,sprd_rtn,prc_op_earnings_basic           |
|      14304 | 0001000187 |        491 | 2021-11-12 00:00:00 |   0.531435 | is_director,prc_op_earnings_basic,ret,price_to_book,sprd_rtn           |
|      11047 | 0001000157 |        255 | 2023-03-16 00:00:00 |   0.52956  | is_director,price_to_book,ret,prc_op_earnings_basic,market_beta        |
|      11896 | 0001000242 |       1039 | 2022-09-23 00:00:00 |   0.525252 | is_director,prc_op_earnings_basic,hml_beta,ret,price_to_book           |
|       2555 | 0001000208 |        457 | 2022-11-09 00:00:00 |   0.521511 | is_director,price_to_book,prc_op_earnings_basic,market_beta,acq_disp_D |
|      13502 | 0001000187 |        491 | 2021-01-20 00:00:00 |   0.518364 | is_director,sprd_rtn,acq_disp_D,market_beta,ret                        |
|        599 | 0001000173 |        456 | 2020-04-21 00:00:00 |   0.50873  | is_director,prc_op_earnings_basic,sprd_rtn,market_beta,price_to_book   |
|       4108 | 0001000096 |        143 | 2022-11-04 00:00:00 |   0.500775 | is_director,market_beta,sprd_rtn,hml_beta,price_to_book                |
|      16692 | 0001000242 |        853 | 2022-09-06 00:00:00 |   0.4969   | is_director,ret,sprd_rtn,hml_beta,price_to_book                        |
|         69 | 0001000170 |        227 | 2022-06-20 00:00:00 |   0.49055  | is_director,sprd_rtn,prc_op_earnings_basic,ret,price_to_book           |
|      20927 | 0001000182 |       1093 | 2022-12-09 00:00:00 |   0.48727  | is_director,ret,price_to_book,acq_disp_D,sprd_rtn                      |
|      13375 | 0001000142 |        341 | 2023-04-18 00:00:00 |   0.485346 | is_director,ret,price_to_book,prc_op_earnings_basic,hml_beta           |
|      28850 | 0001000169 |         13 | 2023-05-01 00:00:00 |   0.484545 | is_director,market_beta,price_to_book,ret,hml_beta                     |
|       4627 | 0001000048 |        780 | 2022-04-28 00:00:00 |   0.478846 | is_director,ret,sprd_rtn,price_to_book,acq_disp_D                      |
|      27670 | 0001000064 |        172 | 2021-12-29 00:00:00 |   0.476464 | is_director,ret,price_to_book,sprd_rtn,acq_disp_D                      |
|       3914 | 0001000193 |        247 | 2021-01-29 00:00:00 |   0.474341 | is_director,ret,acq_disp_D,prc_op_earnings_basic,sprd_rtn              |
|      13783 | 0001000043 |         81 | 2022-04-25 00:00:00 |   0.472726 | is_director,ret,acq_disp_D,prc_op_earnings_basic,market_beta           |
|       6905 | 0001000085 |       1101 | 2023-07-12 00:00:00 |   0.470394 | is_director,ret,sprd_rtn,hml_beta,prc_op_earnings_basic                |
|      15647 | 0001000072 |         33 | 2022-09-06 00:00:00 |   0.46652  | is_director,prc_op_earnings_basic,price_to_book,hml_beta,ret           |
|       4739 | 0001000242 |       1039 | 2022-07-18 00:00:00 |   0.465253 | is_director,hml_beta,ret,price_to_book,sprd_rtn                        |
