# 用python编程实现三种情况的傅里叶反变换  
用python编程实现三种情况的傅里叶反变换  
首先读取了原始图像，并对其进行傅里叶变换。然后，我们构建了一个半径为30的高斯滤波器，并将其与输入图像的频域表示进行卷积。接下来，进行傅里叶反变换，并求得其幅值。最后，将原图像和处理后的图像展示出来差异。  
![%`MH~}QX_H3U5EE)O8ETYNN](https://github.com/H6hh/Fourier-transform/assets/98206033/0e9f263d-4175-4ae3-82bf-26a94a8ea90d)   
当相位谱为0时，傅里叶反变换也称为余弦反变换，我们首先读取了原始图像，并进行了离散余弦变换（DCT）。然后，我们将该图像的相位谱置为0，并进行离散余弦反变换，得到恢复后的图像。最后，我们展示了原始图像和恢复后的图像。  
![image](https://github.com/H6hh/Fourier-transform/assets/98206033/80e47467-37e7-4031-a5bf-941fbc4d4ffa)
在这个例子中，我们使用magnitude_spectrum的一个子窗口，并将其与phase_spectrum组合，以获得所需的幅度谱。我们还通过简单的+运算符将幅度谱转换为复数形式，并将其传递给傅里叶变换的逆变换来获得反变换。注意，我们使用中心位置处的幅度谱子窗口来过滤频域。最后，我们将反变换的结果转换为无符号8位整数（uint8）类型，并将其与原始图像进行比较。   
![image](https://github.com/H6hh/Fourier-transform/assets/98206033/be861e5e-3fad-4aa4-b648-e42a488b2c03)

