# RUN THIS AS THE FINAL CELL IN YOUR EXECUTED GOOGLE COLAB NOTEBOOK.
# It exports the trained branches, fusion head, preprocessing objects and fit-profile
# reference vectors required by the VerityAI backend.
import os, json, joblib, torch, numpy as np

EXPORT_DIR = '/content/verityai_model_artifacts'
os.makedirs(EXPORT_DIR, exist_ok=True)

# Ensure all trained models exist before export.
required = ['attention_model', 'ae_model', 'gcn_model', 'hybrid_model',
            'scaler_dl', 'feature_columns', 'X_fit', 'anomaly_threshold']
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError('Run all model-training cells first. Missing: ' + ', '.join(missing))

for name, model in [('attention', attention_model), ('autoencoder', ae_model),
                    ('gcn', gcn_model), ('fusion', hybrid_model)]:
    model = model.to('cpu').eval()
    torch.save(model.state_dict(), os.path.join(EXPORT_DIR, name + '.pt'))

joblib.dump(scaler_dl, os.path.join(EXPORT_DIR, 'scaler.joblib'))
joblib.dump(list(feature_columns), os.path.join(EXPORT_DIR, 'feature_columns.joblib'))
np.save(os.path.join(EXPORT_DIR, 'fit_reference_vectors.npy'),
        np.asarray(X_fit, dtype=np.float32))

metadata = {
    'project': 'VerityAI - Fake Profile Detection',
    'input_dim': int(len(feature_columns)),
    'feature_columns': list(feature_columns),
    'knn_k': int(K_NEIGHBORS),
    'anomaly_threshold': float(anomaly_threshold),
    'label_mapping': {'0': 'Genuine', '1': 'Fake'},
    'graph_note': 'The research notebook used a transductive graph containing fit, validation and held-out test feature vectors. A live API must construct a query-time graph against stored fit references; this is an inductive serving adaptation and must be separately validated.'
}
with open(os.path.join(EXPORT_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# Also export the best classical baseline for a meaningful comparison.
if 'baseline_results' in globals() and 'best_baseline_name' in globals():
    joblib.dump(baseline_results[best_baseline_name]['model'],
                os.path.join(EXPORT_DIR, 'best_classical_baseline.joblib'))
    metadata['best_classical_baseline'] = str(best_baseline_name)
    with open(os.path.join(EXPORT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

import shutil
archive = shutil.make_archive('/content/VerityAI_Model_Artifacts', 'zip', EXPORT_DIR)
print('Model artifacts exported:', archive)
print('Download this ZIP from the Colab Files panel.')
