# Домашнее задание к уроку 1: Основы PyTorch

## Цель задания
Закрепить навыки работы с тензорами PyTorch, изучить основные операции и научиться решать практические задачи.

## Задание 1: Создание и манипуляции с тензорами

### 1.1 Создание тензоров
```python
Utils.print_header('1.1 Создание тензоров')
Utils.print('Тензор размером 3x4, заполненный случайными числами от 0 до 1',
            torch.rand(3, 4))
Utils.print('Тензор размером 2x3x4, заполненный нулями',
            torch.zeros(2, 3, 4))
Utils.print('Тензор размером 5x5, заполненный единицами',
            torch.ones(5, 5))
Utils.print('Тензор размером 4x4 с числами от 0 до 15',
            torch.reshape(torch.arange(0, 16), (4, 4)))
```

### 1.2 Операции с тензорами
```python
low, high = 0, 10
a = torch.randint(low, high, (3, 4))
b = torch.randint(low, high, (4, 3))
Utils.print('Тензор A', a)
Utils.print('Тензор B', b)
Utils.print('Транспонирование тензора A',
            torch.transpose(a, 0, 1))
Utils.print('Матричное умножение A и B',
            a @ b)
Utils.print('Поэлементное умножение A и транспонированного B',
            a * torch.transpose(b, 0, 1))
Utils.print('Вычислите сумму всех элементов тензора A',
            a.sum())
```

### 1.3 Индексация и срезы
```python
low, high = 0, 10
tensor = torch.randint(low, high, (5, 5))
Utils.print('Тензор',
            tensor)
Utils.print('Первая строка',
            tensor[0])
Utils.print('Последний столбец',
            tensor[:, -1])
Utils.print('Подматрица размером 2x2 из центра тензора',
            tensor[1:4, 1:4])
Utils.print('Все элементы с четными индексами',
            tensor[::2, ::2])
```

### 1.4 Работа с формами
```python
tensor = torch.arange(24)
Utils.print('Тензор', tensor)
Utils.print('2x12', torch.reshape(tensor, (2, 12)))
Utils.print('3x8', torch.reshape(tensor, (3, 8)))
Utils.print('4x6', torch.reshape(tensor, (4, 6)))
Utils.print('2x3x4', torch.reshape(tensor, (2, 3, 4)))
Utils.print('2x2x2x3', torch.reshape(tensor, (2, 2, 2, 3)))
```

## Задание 2: Автоматическое дифференцирование

### 2.1 Простые вычисления с градиентами
```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)
f = x**2 + y**2 + z**2 + 2*x*y*z
f.backward()
x_grad = x.grad
y_grad = y.grad
z_grad = z.grad
Utils.print('Градиент x', x_grad)
Utils.print('Градиент y', y_grad)
Utils.print('Градиент z', z_grad)

# tests
assert x_grad == 14
assert y_grad == 10
assert z_grad == 10
```

### 2.2 Градиент функции потерь
```python
x = torch.tensor(1.0)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

y_pred = w * x + b
y_true = torch.tensor(2.0)
MSE = ((y_pred - y_true) ** 2).mean()
MSE.backward()

Utils.print('Градиент w', w.grad)
Utils.print('Градиент b', b.grad)
```

### 2.3 Цепное правило
```python
x = torch.tensor(1.0, requires_grad=True)
f = (x**2 + 1).sin()
f.backward(retain_graph=True)
x_grad = x.grad
Utils.print('Градиент x', x_grad)

# tests
assert x_grad == torch.autograd.grad(f, x)[0]
```

## Задание 3: Сравнение производительности CPU vs CUDA

### 3.1 Подготовка данных
```python
a = torch.rand(64, 1024, 1024)
b = torch.rand(128, 512, 512)
c = torch.rand(256, 256, 256)
```

### 3.2 Функция измерения времени
```python
def time_it_gpu(self, func: Callable, *args) -> float:
    '''
    Измерить время выполнения функции на GPU
    '''
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record() # type: ignore
    func(*args)
    end.record() # type: ignore
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def time_it_cpu(self, func: Callable, *args) -> float:
    '''
    Измерить время выполнения функции на CPU
    '''
    start = time.time()
    func(*args)
    end = time.time()
    return (end - start) * 1000
```

### 3.3 Сравнение операций
| Операция                          | CPU (мс) | GPU (мс) | Ускорение |
|-----------------------------------|----------|----------|-----------|
| Матричное умножение A на A        | 351.90   | 110.36   | 3.19      |
| Матричное умножение B на B        | 134.01   | 7.39     | 18.12     |
| Матричное умножение C на C        | 30.00    | 2.26     | 13.27     |
| Поэлементное сложение A и A       | 55.31    | 19.50    | 2.84      |
| Поэлементное сложение B и B       | 22.00    | 2.32     | 9.47      |
| Поэлементное сложение С и C       | 6.53     | 1.17     | 5.57      |
| Поэлементное умножение A на A     | 62.07    | 11.72    | 5.30      |
| Поэлементное умножение B на B     | 15.51    | 2.32     | 6.69      |
| Поэлементное умножение C на C     | 6.00     | 1.16     | 5.18      |
| Транспонирование A                | 0.00     | 0.07     | 0.00      |
| Транспонирование B                | 0.00     | 0.03     | 0.00      |
| Транспонирование C                | 0.00     | 0.02     | 0.00      |
| Вычисление суммы всех элементов A | 6.00     | 10.97    | 0.55      |
| Вычисление суммы всех элементов B | 3.01     | 0.81     | 3.70      |
| Вычисление суммы всех элементов C | 2.00     | 0.43     | 4.62      |

### 3.4 Анализ результатов
Можно заметить, что выполнение на GPU дает существенную скорость на некоторых задачах. Скорость зависила от количества набора данных и сложности выполнения операции.
Но хочется больше уделить объяснению того, почему на транспонирование данных практически не затратилось время? Это связано с тем, что операция транспонирования относится к представлениям (views), суть которых заключается в том, что они работают с метаданными, благодаря чему мы избегаем явного копирования. [Источник с официальной документации](https://docs.pytorch.org/docs/stable/tensor_view.html)
