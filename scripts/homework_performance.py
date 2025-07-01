import torch
import time
from typing import Callable
from prettytable import PrettyTable


class Profiler:
    def __init__(self) -> None:
        self.table = PrettyTable(['Операция', ' CPU (мс)', 'GPU (мс)', 'Ускорение'], float_format='.2')
    
    def stats(self) -> None:
        '''
        Вывод таблицы с результатами
        '''
        print(self.table)

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
    
    def test(self, testname: str, func: Callable, *args) -> None:
        '''
        Запустить профилирование
        '''
        # подготавливаем данные для теста на GPU
        args_gpu = [arg.cuda() if isinstance(arg, torch.Tensor) else arg for arg in args]
        # подготавливаем данные для теста на CPU
        args_cpu = [arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args]

        cpu_time = self.time_it_cpu(func, *args_cpu)
        gpu_time = self.time_it_gpu(func, *args_gpu)
        boost = cpu_time / gpu_time

        self.table.add_row([testname, cpu_time, gpu_time, boost])


a = torch.rand(64, 1024, 1024)
b = torch.rand(128, 512, 512)
c = torch.rand(256, 256, 256)

profiler = Profiler()
profiler.test('Матричное умножение A на A', torch.matmul, a, a)
profiler.test('Матричное умножение B на B', torch.matmul, b, b)
profiler.test('Матричное умножение C на C', torch.matmul, c, c)
profiler.test('Поэлементное сложение A и A', lambda t1, t2: t1 + t2, a, a)
profiler.test('Поэлементное сложение B и B', lambda t1, t2: t1 + t2, b, b)
profiler.test('Поэлементное сложение С и C', lambda t1, t2: t1 + t2, c, c)
profiler.test('Поэлементное умножение A на A', lambda t1, t2: t1 * t2, a, a)
profiler.test('Поэлементное умножение B на B', lambda t1, t2: t1 * t2, b, b)
profiler.test('Поэлементное умножение C на C', lambda t1, t2: t1 * t2, c, c)
profiler.test('Транспонирование A', lambda t: t.T, a)
profiler.test('Транспонирование B', lambda t: t.T, b)
profiler.test('Транспонирование C', lambda t: t.T, c)
profiler.test('Вычисление суммы всех элементов A', lambda t: t.sum(), a)
profiler.test('Вычисление суммы всех элементов B', lambda t: t.sum(), b)
profiler.test('Вычисление суммы всех элементов C', lambda t: t.sum(), c)

profiler.stats()
