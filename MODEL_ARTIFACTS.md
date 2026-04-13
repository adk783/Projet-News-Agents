# Model artifacts

This repository now includes the inference artifacts needed to run the enriched scoring pipeline.

## Included weights

- `uncertainty_model/`: full uncertainty model for `UncertaintyAgent`.
- `litigious_model/model.joblib`: TF-IDF + Ridge model for `LitigiousAgent`.
- `fundamental_strength_model/model.joblib`: TF-IDF + Ridge model for `FundamentalStrengthAgent`.
- `market_impact_model/classifier.joblib`: final hybrid classifier for `Bullish / Neutral / Bearish`.
- `market_impact_model/kmeans.joblib`: auxiliary KMeans model used by the final classifier.

`uncertainty_model/model.safetensors` is stored with Git LFS because it is too large for normal Git storage.

## Output weights

Run:

```bash
python print_output_weights.py
```

This prints and exports the numeric weights of the base outputs in the final model:

- `polarity`
- `polarity_conf`
- `uncertainty`
- `litigious`
- `fundamental_strength`

Generated CSV files are written to `metrics_plots/`.
