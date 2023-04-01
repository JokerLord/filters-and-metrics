# Фильтры и метрики
___
## Программа реализует основные алгоритмы фильтрации изображения и метрики:
### Вычислить значение метрики MSE и вывести его на консоль. Запуск:
    python main.py mse (input_file_1) (input_file_2)
### Вычислить значение метрики PSNR и вывести . Запуск:
    python main.py psnr (input_file_1) (input_file_2)
### . Запуск:
    main.py dec -k sec.key input.txt.enc -o output1.txt
### Создание модели (возвращает массив из частот встречаемости каждого байта на основании большого кол-ва текствов). Запуск:
    main.py makemodel [file1.txt, file2.txt, ...] -o model.txt
### Дешифрование (без ключа) (возвращает ключ). Запуск:
    main.py broke model.txt input.txt -o sec1.key
