
FAKE PROFILE DETECTION USING A HYBRID DEEP LEARNING FRAMEWORK

Project components:
1. Classical ML baselines: Logistic Regression, Random Forest, SVM, XGBoost.
2. Feature-attention neural network.
3. Autoencoder-based anomaly scoring trained on genuine training profiles.
4. Graph convolution network using a k-nearest-neighbour similarity graph.
5. Hybrid fusion classifier.
6. SHAP and attention-based explanations.
7. Fusion-level ablation study.
8. Explainable profile evidence and similar-profile lookup.

Dataset:
696 labelled profiles, with supplied train and test partitions.
The dataset contains 11 original predictors. Five derived features are added.
Total model input: 16 features.

Important evaluation notes:
- The supplied test labels are held out from supervised model training.
- The graph experiment is transductive: test feature vectors are included
  when constructing the similarity graph, but test labels are not used.
- The autoencoder is trained using genuine training profiles only.
- SHAP explains the learned fused representation; attention weights provide
  an additional view of original input features.
- The ablation study compares fusion heads over learned branch embeddings.
- Results are experimental and require external validation on newer data.

Run the notebook from top to bottom in Google Colab.
