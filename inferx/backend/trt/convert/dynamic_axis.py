class DynamicAxisInfo:
    def __init__(self, name: str, min_shape: tuple, opt_shape: tuple, max_shape: tuple):
        """
        封装动态轴信息
        :param name: 输入张量的名称
        :param min_shape: 最小输入形状
        :param opt_shape: 优化输入形状
        :param max_shape: 最大输入形状
        """
        self.__name = name  # 私有属性，输入张量的名称
        self.__min_shape = min_shape  # 私有属性，动态输入的最小形状
        self.__opt_shape = opt_shape  # 私有属性，动态输入的优化形状
        self.__max_shape = max_shape  # 私有属性，动态输入的最大形状

    @property
    def input_name(self) -> str:
        """获取输入张量的名称"""
        return self.__name

    @property
    def min_shape(self) -> tuple:
        """获取动态输入的最小形状"""
        return self.__min_shape

    @property
    def opt_shape(self) -> tuple:
        """获取动态输入的优化形状"""
        return self.__opt_shape

    @property
    def max_shape(self) -> tuple:
        """获取动态输入的最大形状"""
        return self.__max_shape
