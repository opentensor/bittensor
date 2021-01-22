class Balance:
    unit = "Tao"
    rao : int
    tao: float

    # Balance should always be an int and represent the satoshi of our token:rao
    def __init__(self, balance):
        self.rao = balance
        self.tao = self.rao / pow(10, 9)

    def __int__(self):
        return self.rao

    def __float__(self):
        return self.tao

    def __str__(self):
        return "{unit:s} {balance:.9f}".format(unit=self.unit, balance=self.tao)

    def __eq__(self, other):
        return self.rao == other.rao

    def __ne__(self, other):
        return self.rao != other.rao

    def __gt__(self, other):
        return self.rao > other.rao

    def __lt__(self, other):
        return self.rao < other.rao

    def __le__(self, other):
        return self.rao <= other.rao

    def __ge__(self, other):
        return self.rao >= other.rao

    @staticmethod
    def from_float(amount : float):
        rao = int(amount * pow(10, 9))
        return Balance(rao)


