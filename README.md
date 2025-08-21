# CNN-LSTM Weather Forecast Model

![Python CI](https://github.com/jack-ianson/forecast-cnn-lstm/actions/workflows/ci.yml/badge.svg)

A Convolutional neural network (CNN) coupled with a Long Short Term Memory (LSTM) network for predicting the temperature data one hour into the future based on the previous 24 hours of weather data, including temperature, wind speed, cloud cover, etc.

The source data was gathered from Open-Meteo (https://open-meteo.com/) using a 0.5 degree grid with a shape of 28x28 pixels (current low resolution for testing). The target window was 24x24 pixels, which allows the CNN-LSTM to "see" an extra 2 data points in each direction, improving the prediction accuracy.

An example of the prediction compared to the true temperature is shown below.

![alt text](/images/image_1.png)
