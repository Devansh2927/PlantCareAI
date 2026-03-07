# PlantCareAI

PlantCareAI uses transfer learning (MobileNetV2) to classify plant leaf diseases and provide care recommendations.
PlantCare AI is a deep learning-based solution designed to identify 38 different classes of plant diseases from leaf images. By leveraging MobileNetV2 and Transfer Learning, this project provides a lightweight yet highly accurate tool for farmers and gardeners to monitor crop health in real-time.

Key Features
High Accuracy: Achieved 96.6% validation accuracy using the New Plant Diseases Dataset.

Transfer Learning: Utilizes pre-trained MobileNetV2 weights for faster convergence and lower computational costs.

Real-time Detection: Integrated with a Flask web interface for instant image analysis.

Scalable: Designed for deployment in automated farm monitoring or mobile apps.

Dataset
The model is trained on the New Plant Diseases Dataset, which contains:

87,000+ images of healthy and diseased plant leaves.

38 distinct classes including crops like Apple, Corn, Grape, Potato, and Tomato.

Annotated images categorized by species and specific pathology.
