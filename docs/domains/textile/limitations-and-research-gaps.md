# Limitations and Research Gaps

1. **Unconfirmed Target Semantics:** Both `weaving` (`Rej_and_cut_Piece`) and `knitting` (`Reject Qty`) remain `TargetConfirmation.UNCONFIRMED` in production contracts pending factory industrial sign-off. Production training is intentionally blocked unless explicitly confirmed.
2. **Weaving Missing Calendar Dates:** Real weaving data only contains month strings without year or day; temporal splitting is infeasible, requiring `SplitStrategy.GROUP` on `order_id`.
3. **Knitting Missing `run_id` & Structural Sparsity:** Real knitting exports lack an explicit `run_id` column and exhibit 61.24% / 92.41% structural missingness in `yarn_count_tex_2` / `yarn_count_tex_3`.
4. **Physical Anomalies in Historical Logs:** 140 weaving rows have rejection exceeding planned beam length; 10 knitting rows have rejection exceeding 100% of shift total.
5. **Exploratory Scope Only:** Real dataset experiments must remain exploratory until target definitions, unit provenance, and factory boundaries are formally confirmed.
