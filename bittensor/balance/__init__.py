class Balance:
    unit = "Tao"
    rao : int
    tao: float

    def __init__(self, balance):
        self.rao = balance
        self.tao = self.rao / pow(10, 9)

    def __int__(self):
        return self.rao

    def __float__(self):
        return self.tao

    def __str__(self):
        return "{unit:s} {balance:.9f}".format(unit=self.unit, balance=self.tao)
