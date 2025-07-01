import torch
from utils import Utils


class Tasks:
    @staticmethod
    def run_2_1() -> None:
        '''
        Запуск задания 2.1
        '''
        Utils.print_header('2.1 Простые вычисления с градиентами')
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
    
    @staticmethod
    def run_2_2() -> None:
        '''
        Запуск задания 2.2
        '''
        Utils.print_header('2.2 Градиент функции потерь')

        x = torch.tensor(1.0)
        w = torch.tensor(0.5, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)

        y_pred = w * x + b
        y_true = torch.tensor(2.0)
        MSE = ((y_pred - y_true) ** 2).mean()
        MSE.backward()

        Utils.print('Градиент w', w.grad)
        Utils.print('Градиент b', b.grad)

    @staticmethod
    def run_2_3() -> None:
        '''
        Запуск задания 2.3
        '''
        Utils.print_header('2.3 Цепное правило')

        x = torch.tensor(1.0, requires_grad=True)
        f = (x**2 + 1).sin()
        f.backward(retain_graph=True)
        x_grad = x.grad
        Utils.print('Градиент x', x_grad)

        # tests
        assert x_grad == torch.autograd.grad(f, x)[0]

    @staticmethod
    def run_all() -> None:
        '''
        Запуск всех заданий
        '''
        Tasks.run_2_1()
        Tasks.run_2_2()
        Tasks.run_2_3()


if __name__ == '__main__':
    Tasks.run_all()
