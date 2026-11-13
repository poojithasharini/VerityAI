# Connecting the executed notebook to VerityAI

## Why this is required
A `.ipynb` file is an experiment document, not a prediction service. The browser cannot directly execute its Python/PyTorch cells. The notebook trains the model; an export cell saves trained weights and preprocessing objects; FastAPI loads them and exposes a prediction endpoint; React calls that endpoint.

## Export cell
Use `EXPORT_NOTEBOOK_MODELS.py` as the last cell of the executed notebook. The companion notebook `capstone_verityai_export_ready.ipynb` already contains that cell appended. Run the entire notebook from top to bottom, then run the final cell. It creates `VerityAI_Model_Artifacts.zip` in Colab.

The artifact ZIP includes:
- `attention.pt`, `autoencoder.pt`, `gcn.pt`, `fusion.pt`
- `scaler.joblib`, `feature_columns.joblib`
- `fit_reference_vectors.npy`
- `metadata.json` (input dimension, feature order, KNN K, anomaly threshold and graph caveat)
- `best_classical_baseline.joblib` when the notebook's baseline objects are present

Extract the contents to `backend/model_artifacts/` (create that folder if needed).

## Important inference caveat
The notebook's GCN was trained/evaluated on one transductive graph assembled from fit, validation and test feature vectors. Test labels were not used, but test feature vectors were present in graph construction. A live user profile is not part of that original graph. For serving, the backend must connect a new profile to stored fit-reference nodes, rebuild the normalized adjacency for that query, and pass the resulting embeddings through the saved fusion head. This is an inductive serving adaptation, not exactly the same inference condition as the notebook test. Re-evaluate it on a held-out protocol before presenting deployment metrics as validated.

## Feature mapping for the form
The notebook uses 16 ordered features: the 11 source predictors plus `follower_following_ratio`, `log_followers`, `log_following`, `log_posts`, and `profile_completeness`. Raw username/full-name/bio text is not present in the supplied dataset; do not invent text-overlap features. `fake=0` means genuine and `fake=1` means fake in the dataset.

## Security and responsible use
The project is a research prototype using a small, older, labelled dataset. Profile metadata alone cannot establish identity or intent. Show outputs as review signals, not definitive accusations or automatic enforcement decisions. Avoid collecting Instagram credentials or scraping private accounts.
