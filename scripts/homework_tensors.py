import torch
from utils import Utils


class Tasks:
    @staticmethod
    def run_1_1() -> None:
        '''
        Запуск задания 1.1
        '''
        Utils.print_header('1.1 Создание тензоров')
        Utils.print('Тензор размером 3x4, заполненный случайными числами от 0 до 1',
                    torch.rand(3, 4))
        Utils.print('Тензор размером 2x3x4, заполненный нулями',
                    torch.zeros(2, 3, 4))
        Utils.print('Тензор размером 5x5, заполненный единицами',
                    torch.ones(5, 5))
        Utils.print('Тензор размером 4x4 с числами от 0 до 15',
                    torch.reshape(torch.arange(0, 16), (4, 4)))
    
    @staticmethod
    def run_1_2() -> None:
        '''
        Запуск задания 1.2
        '''
        Utils.print_header('1.2 Операции с тензорами')
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
    
    @staticmethod
    def run_1_3() -> None:
        '''
        Запуск задания 1.3
        '''
        Utils.print_header('1.3 Индексация и срезы')
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
    
    @staticmethod
    def run_1_4() -> None:
        '''
        Запуск задания 1.4
        '''
        Utils.print_header('1.4 Работа с формами')
        tensor = torch.arange(24)
        Utils.print('Тензор', tensor)
        Utils.print('2x12', torch.reshape(tensor, (2, 12)))
        Utils.print('3x8', torch.reshape(tensor, (3, 8)))
        Utils.print('4x6', torch.reshape(tensor, (4, 6)))
        Utils.print('2x3x4', torch.reshape(tensor, (2, 3, 4)))
        Utils.print('2x2x2x3', torch.reshape(tensor, (2, 2, 2, 3)))

    @staticmethod
    def run_all() -> None:
        '''
        Запуск всех заданий
        '''
        Tasks.run_1_1()
        Tasks.run_1_2()
        Tasks.run_1_3()
        Tasks.run_1_4()


if __name__ == '__main__':
    Tasks.run_all()
