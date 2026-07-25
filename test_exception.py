import sys

from src.exception import CustomException


def divide_numbers():
    try:
        a = 10
        b = 0

        print(a / b)

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    divide_numbers()