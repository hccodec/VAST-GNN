from datetime import date, datetime, timedelta

class datetime(datetime):
    def __add__(self, other):
        if isinstance(other, int): return self + timedelta(days=other)
        else: return super().__add__(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, int): return self + timedelta(days=-other)
        else:
            return super().__sub__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

class date(date):
    def __add__(self, other):
        if isinstance(other, int): return self + timedelta(days=other)
        else: return super().__add__(other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, int): return self + timedelta(days=-other)
        else:
            return super().__sub__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

def str2date(f): return f if isinstance(f, datetime) else datetime.strptime(f, '%Y%m%d')
def date2str(f): return f if isinstance(f, str) else datetime.strftime(f, '%Y%m%d')