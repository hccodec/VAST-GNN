from datetime import datetime, timedelta

class timedelta(timedelta):
    def __add__(self, other):
        if isinstance(other, int):
            return self + timedelta(days=other)
        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            return self - timedelta(days=other)
        return super().__sub__(other)

    def __mul__(self, other):
        if isinstance(other, int):
            return timedelta(seconds=self.total_seconds() * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, int):
            return timedelta(seconds=self.total_seconds() / other)
        return NotImplemented

class datetime(datetime):
    """
    自定义的 datetime 类，继承自内置的 datetime 类，并重载了一些运算符方法。

    方法:
        __add__(self, other):
            重载加法运算符。如果 other 是整数，则返回当前日期加上 other 天数后的日期。
            否则，调用父类的 __add__ 方法。

        __sub__(self, other):
            重载减法运算符。如果 other 是整数，则返回当前日期减去 other 天数后的日期。
            否则，调用父类的 __sub__ 方法。

        __mul__(self, other):
            重载乘法运算符。如果 other 是整数，则返回从1970年1月1日开始，当前日期与1970年1月1日的时间差乘以 other 后的日期。
            否则，返回 NotImplemented。

        __truediv__(self, other):
            重载除法运算符。如果 other 是整数，则返回从1970年1月1日开始，当前日期与1970年1月1日的时间差除以 other 后的日期。
            否则，返回 NotImplemented。
    """
    def __add__(self, other):
        if isinstance(other, int):
            return self + timedelta(days=other)
        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, int):
            return self - timedelta(days=other)
        return super().__sub__(other)

    def __mul__(self, other):
        if isinstance(other, int):
            delta = self - datetime(1970, 1, 1)
            return datetime(1970, 1, 1) + timedelta(seconds=delta.total_seconds() * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, int):
            delta = self - datetime(1970, 1, 1)
            return datetime(1970, 1, 1) + timedelta(seconds=delta.total_seconds() / other)
        return NotImplemented

def str2date(s: str, fmt: str = None) -> datetime:
    if fmt:
        return datetime.strptime(s, fmt)
    
    # 支持的日期格式列表
    supported_formats = [
        '%Y/%m/%d',  # 2023/10/05
        '%Y-%m-%d',  # 2023-10-05
        '%d/%m/%Y',  # 05/10/2023
        '%d-%m-%Y',  # 05-10-2023
        '%Y%m%d',    # 20231005 (无分隔符)
    ]
    
    # 尝试每种格式
    for fmt in supported_formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    
    # 如果所有格式都失败，抛出异常
    raise ValueError(f"无法解析日期字符串: {s}")

def date2str(d: datetime, fmt = '%Y%m%d'):
    return datetime.strftime(d, fmt)

class DateRange:
    def __init__(self, start, end, step=1):
        if isinstance(start, str): start = str2date(start)
        if isinstance(end, str): end = str2date(end)
        assert isinstance(start, datetime) and isinstance(end, datetime)
        self.start, self.end = start, end
        self.step = timedelta(days=step) if isinstance(step, int) else step
        self.current = start

    def __iter__(self):
        self.current = self.start
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        current = self.current
        self.current += self.step
        return current
    
    def __len__(self):
        delta = self.end - self.start
        step_in_days = self.step.days
        return (delta.days + step_in_days - 1) // step_in_days
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self.start + i * self.step for i in range(start, stop, step)]
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index >= len(self) or index < 0:
                raise IndexError("DataRange index out of range")
            return self.start + index * self.step
        else:
            raise TypeError("DataRange indices must be integers or slices, not {}".format(type(index).__name__))

def continuous(lis):
    '''
    分析一个日期列表的不连续处
    '''
    
    lis_date = [str2date(i) for i in lis]
    res = []
    for i in range(len(lis) - 1):
        if (lis_date[i + 1] - lis_date[i]).days == 1: continue
        res.append([lis[i], lis[i + 1]])
    return res
