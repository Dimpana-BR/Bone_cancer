# TODO: Improve Quantum SVM Accuracy

## Completed Tasks
- [x] Analyze current quantum SVM setup and identify issues (fallback to classical SVC, small dataset, poor feature selection, default parameters).

## Pending Tasks
- [ ] Update main.py: Increase QUANTUM_SAMPLES to 50 and QUANTUM_FEATURES to 5 for more training data.
- [ ] Update quantum_preprocess.py: Implement better feature selection (e.g., select features with highest variance).
- [ ] Update quantum_svm_qiskit.py: Tune QSVC parameters (increase reps to 3) and enhance fallback SVC (use 'poly' kernel, C=1.0).
- [ ] Test changes by running main.py and verify accuracy improvement.
- [ ] If needed, further tune parameters or increase data size.
