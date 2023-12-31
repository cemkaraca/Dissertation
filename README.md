# Turbulence Forecasting with Neural Networks
## Abstract
Turbulence remains one of the most challenging problems in physics, impacting various engineering designs from energy to aerospace. This repository includes the code for a model that leverages machine learning techniques, specifically two neural network architectures: Long-Short Term Memory (LSTM) and an LSTM Encoder-Decoder-based model with an attention mechanism.

### Motivation
A better understanding of turbulence is crucial for enhancing the quality of design and optimization of engineering systems exposed to turbulent flows. Traditional methods, such as well-resolved numerical simulations, are computationally expensive. This project explores the use of machine learning to predict turbulence, offering a promising alternative by reducing computational costs while maintaining reasonable accuracy.

### Model Overview
The neural networks are trained using data samples from turbulent flows obtained through high-fidelity numerical simulations. The turbulent flows, known for their chaotic and correlated nature, pose challenges for mathematical representation. The LSTM and LSTM Encoder-Decoder models effectively capture the complex, non-linear relationships in turbulent flow fields, making them valuable tools for forecasting turbulence time series

### Results
The forecasts generated by the models are compared with reference Computational Fluid Dynamics (CFD) data. The results show satisfactory performance with reduced computational complexity compared to traditional simulation methods. However, limitations exist, such as the LSTM model's potential for overfitting and the encoder-decoder-based model's inconsistent predictions across the flow.

### Future Work
Future work could explore advanced techniques such as PINNs(Physics Informed Neural Networkds) or transfer learning to improve model accuracy and generalization capabilities. These enhancements aim to address the identified limitations and further advance the application of machine learning in turbulence prediction.

## Repository Structure

- **`/code`**: Contains the implementation of the LSTM and LSTM Encoder-Decoder models.
- **`/data`**: Placeholder for the dataset used in training and testing the models.
