# Фильтры и метрики
___
## Программа реализует основные алгоритмы фильтрации изображения и метрики:
### Вычислить значение метрики MSE и вывести его на консоль. Запуск:
    python main.py mse (input_file_1) (input_file_2)
### Вычислить значение метрики PSNR и вывести. Запуск:
    python main.py psnr (input_file_1) (input_file_2)
### Вычислить значение метрики SSIM и вывести. Запуск:
    python main.py ssim (input_file_1) (input_file_2)
### Медианная фильтрация с окном размера (2rad+1) &times; (2rad+1). Запуск:
    python main.py median (rad) (input_file) (output_file)
### Фильтр Гаусса с параметром $\sigma_d$. Запуск:
    python main.py gauss (sigma_d) (input_file) (output_file)
### Билатеральная фильтрация с параметрами $\sigma_d$ и $\sigma_r$. Запуск:
    python main.py bilateral (sigma_d) (sigma_r) (input_file) (output_file)
Значение rad - целое положительное, значения sigma_d и sigma_r - вещественные положительные.
