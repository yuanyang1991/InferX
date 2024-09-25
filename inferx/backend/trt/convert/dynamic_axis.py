from typing import Tuple


class DynamicAxisInfo:
    def __init__(self, name: str, min_value: Tuple, opt_value: Tuple, max_value: Tuple):
        """
        封装动态轴信息
        :param name: 输入张量的名称
        :param min_value: 最小输入形状
        :param opt_value: 优化输入形状
        :param max_value: 最大输入形状
        """
        self.__name = name  # 私有属性，输入张量的名称
        self.__min = min_value  # 私有属性，动态输入的最小形状
        self.__opt = opt_value  # 私有属性，动态输入的优化形状
        self.__max = max_value  # 私有属性，动态输入的最大形状

    @property
    def input_name(self) -> str:
        """获取输入张量的名称"""
        return self.__name

    @property
    def min_shape(self) -> Tuple:
        """获取动态输入的最小形状"""
        return self.__min

    @property
    def opt_shape(self) -> Tuple:
        """获取动态输入的优化形状"""
        return self.__opt

    @property
    def max_shape(self) -> Tuple:
        """获取动态输入的最大形状"""
        return self.__max
