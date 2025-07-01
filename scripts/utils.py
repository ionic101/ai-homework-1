from typing import Iterable


class Utils:
    @staticmethod
    def print(message: str, *objects: Iterable | None) -> None:
        '''
        Красивый вывод
        '''
        objects = objects if objects else ()
        print(message, '-' * len(message), *objects, sep='\n', end='\n\n')
    
    @staticmethod
    def print_header(header: str) -> None:
        '''
        Вывод заголовков
        '''
        line = '=' * len(header)
        print(line, header, line, sep='\n', end='\n\n')
