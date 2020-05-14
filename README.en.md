# ClassifierEstimator

#### Description
分类器评估器：属于解释性AI的一个探索，通过对输入样本在已学习的样本空间中进行分解和重组，观测该样本和已学习的样本分布的服从度（当该样本采样自已学习的样本分布时，则服从度高，否则服从度低），据此，如果从训练样本空间以外（比如不同的数据集不同的领域）采样输入，则评估器会评估该样本的服从度，该服从度直接视为分类器输出结果的置信度。
ClassifierEstimator: this research belongs to the so called "Interpretable AI", which attempts to decompose and rebuild the input unknown sample, and observe the obedience of the sample to the trained sample space. If the obedience is high, then it indicates that this sample is within known domain, and the prediction should be of high confidence. Otherwise, it indicates this sample comes from unknown domain, the predicted result is under suspection.