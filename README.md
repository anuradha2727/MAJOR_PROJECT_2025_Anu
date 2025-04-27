Machine Learning-based Intrusion Detection System using CIC-IDS 2017 dataset with feature selection, model optimization, and attack classification.
Modern cyber threats are becoming more complex, making it harder for traditional Intrusion Detection Systems (IDS) to detect them effectively. Traditional IDS systems are increasingly inadequate, particularly in identifying sophisticated and evolving attacks. This creates a need for efficient systems that can not only classify attacks accurately but also minimize false positives, which are a significant issue in real-world applications.

To address these challenges, this project develops a machine learning-based IDS that can effectively classify and detect network intrusions with low false positives. The CIC-IDS2017 dataset was selected because it simulates realistic network traffic and has various types of attacks, such as DoS, DDoS, Brute Force, SQL Injection, and Botnet attacks.

The methodology involved extensive data preprocessing, which included missing value handling, removing duplicates, categorical encoding, and Random Forest feature selection. The dataset was balanced using SMOTE (Synthetic Minority Over-sampling Technique). Hyperparameter optimization was also performed for the Random Forest classifier using the Firefly Algorithm, resulting in a notable enhancement in model performance.

The model was tested with 10, 15, 20, 30 and 40 selected features, with post-optimization accuracy reaching 99.45% for the 30-feature model. Evaluation metrics such as accuracy, detection rate, precision, recall, F1-score, and false positive rate confirmed the effectiveness and scalability of the proposed IDS. This optimized approach enhances cyber threat detection and provides a robust, real-world cybersecurity solution.

Dataset link: https://www.dropbox.com/scl/fi/0vfye4bksapb1qglmq989/archive.zip?rlkey=mc5nellna1gqalho86ktk65zm&st=q9dp0lzb&dl=0

Download the archieve file which contains the dataset files from the above link.
